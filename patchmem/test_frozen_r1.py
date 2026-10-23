"""PAT-307 frozen-R1 checks (ETTh2, A1 seed 3047 checkpoint): for O, ErrPath and T0r with R1_FREEZE_FROM
(1) before training the test MSE at 1440/336 == the A1 log (.433825) (zero-init module + the A1 backbone);
(2) only module parameters require grad; (3) a short training run (R1_MAX_ITERS) leaves every backbone tensor bit-identical
(asserted inside Exp_R1.train) and changes the module; (4) the backbone sub-modules run in eval mode during training and every training forward scans in the fixed eval order.
Eval order: R1_EVAL_PERM / R1_ATSP_SEED unset, as in the A1 run.
usage: python patchmem/test_frozen_r1.py"""
import os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from patchmem.diag_rollout import official_args, ETTH2

A1 = 'checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_R1h0s3047_0/checkpoint.pth'
ARMS = {'O': dict(R1_OBS='1', R1_MEM_QK_DIM='8', R1_MEM_SLOT_KEEP='0.5'),
        'EP': dict(R1_OBS='1', R1_OBS_VPATH='1', R1_MEM_QK_DIM='8', R1_MEM_SLOT_KEEP='0.5'),
        'T0r': dict(R1_TEFL='native', R1_TEFL_HC='2', R1_TEFL_RANK='64')}
KEYS = ('R1_OBS', 'R1_OBS_VPATH', 'R1_MEM_QK_DIM', 'R1_MEM_SLOT_KEEP', 'R1_TEFL', 'R1_TEFL_HC', 'R1_TEFL_RANK')
from patchmem.exp_r1 import Exp_R1
for arm, env in ARMS.items():
    for k in KEYS + ('R1_EVAL_PERM', 'R1_ATSP_SEED'):
        os.environ.pop(k, None)
    os.environ.update(R1_FREEZE_FROM=A1, R1_MAX_ITERS='30', R1_VAL_STRIDE='64', R1_AUDIT='0', **env)
    a = official_args(ETTH2 + ['--des', f'frozentest{arm}', '--seed', '3047', '--is_training', '1', '--train_epochs', '1'])
    ex = Exp_R1(a); m = ex.model
    mse = ex.eval_grid('test', [(1440, 336)])[(1440, 336)]['mse']
    print(f'{arm} 1 init test 1440/336 {mse:.6f} vs A1 .433825'); assert abs(mse - 0.433825) < 2e-6, mse
    tr = [n for n, p in m.named_parameters() if p.requires_grad]
    assert tr and all(n.startswith(('obs_net.', 'W1.', 'W2.')) for n in tr), tr
    print(f'{arm} 2 trainable: {len(tr)} tensors, all in the module')
    mod0 = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    modes = []
    order = ex.eval_order(); perms = []
    so = m.scan_order
    m.scan_order = lambda b, d: (lambda p: (perms.append(p.clone()), p)[1])(so(b, d))
    orig = m.read
    m.read = lambda *x_, **k_: (modes.append((m.training, m.base.training, m.readout.training)), orig(*x_, **k_))[1]
    setting = f'frozentest_{arm}'
    ex.train(setting)
    m.read = orig; m.scan_order = so
    assert perms and all((p_ == torch.as_tensor(order, device=p_.device)).all() for p_ in perms), 'training scan order not fixed'
    print(f'{arm} 4b all {len(perms)} forwards scanned in the eval order {order}')
    tm = [t for t in modes if t[0]]
    assert tm and all(not b and not r for _, b, r in tm), tm[:3]
    print(f'{arm} 4 during training: R1 training True, base/readout eval ({len(tm)} reads)')
    ch = max((p.detach() - mod0[n]).abs().max().item() for n, p in m.named_parameters() if n in mod0)
    print(f'{arm} 3 module changed by {ch:.2e}; backbone unchanged (asserted in train)'); assert ch > 0
print('ALL OK')
