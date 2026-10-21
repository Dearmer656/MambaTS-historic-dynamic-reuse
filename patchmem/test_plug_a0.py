"""PAT-307 plug-in checks (patchmem/plug_a0.py) on the A0 ETTh2 L720/H96 seed-3047 checkpoint and its cache:
(1) causality: every observed residual entry reads x at an index < t; every cached forecast used is issued at an origin <= t;
(2) the cached forecasts / residuals equal a direct forward of A0 on the corresponding raw windows (independent of the cache);
    TEFL's eps (origin m* = H/P) == x[t-H+k] - f(x[t-H-L : t-H])[k] normalised by the current lookback std;
(3) every plug-in == base at init; the loss reaches the plug-in parameters; the 'zero' audit == base."""
import sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args
from patchmem.plug_a0 import load_a0, SplitData, Plugin

A = ('--task_name long_term_forecast --is_training 0 --root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 --dec_in 7 --c_out 7 '
     '--model_id ETTh2_720_96 --model MambaTS --data ETTh2 --features M --seed 3047 --seq_len 720 --label_len 48 --pred_len 96 --e_layers 5 '
     '--d_layers 2 --factor 1 --des ExpA0s3047 --itr 1 --n_heads 16 --d_model 16 --dropout 0.3 --patch_len 48 --stride 48 --VPT_mode 1 '
     '--ATSP_solver SA').split()
S = 'long_term_forecast_ETTh2_720_96_MambaTS_ETTh2_ftM_sl720_ll48_pl96_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_ExpA0s3047_0'
args = official_args(A)
L, H, P, M = 720, 96, 48, 15
m, dev, order = load_a0(args, S, 3047)
C = torch.load(f'patchmem/logs/plug_cache/{S}_M{M}_v2.pt', weights_only=False)
sd_ = SplitData(C['test']['x'], C['test']['Y'], C['test']['Z'], L, H, M, P)
idx = [0, 1, 777, sd_.n - 1]
d = sd_.batch(idx, dev)
x = sd_.x
t = torch.as_tensor(idx) + sd_.Lext
# (1) causality (index level)
s = t[:, None] - torch.arange(1, M + 1)[None, :] * P
tidx = s[:, :, None] + torch.arange(H)[None, None, :]
assert bool((tidx[d['obs'].cpu()] < t[:, None, None].expand_as(tidx)[d['obs'].cpu()]).all())
assert bool((s <= t[:, None]).all()) and bool((s - L >= 0).all())
assert bool(((d['R'] != 0).cpu() <= d['obs'].cpu()[..., None]).all()), 'unobserved residual entries must be 0'
print('(1) causality: observed residuals read x[< t]; origins <= t; unobserved entries are 0')
# (2) direct A0 forward on raw windows
with torch.no_grad():
    for j, i in enumerate(idx):
        tt = int(t[j])
        for mm in (0, 1, 2, 7):
            ss = tt - mm * P
            bx = x[ss - L:ss][None].to(dev)
            m.encoder.ids_shuffle = order.repeat(1, 1)
            y = m(bx, None, None, None)[0].cpu()
            ref_y = C['test']['Y'][ss - L]
            assert (y - ref_y).abs().max() < 1e-4, (i, mm, (y - ref_y).abs().max())
        look = x[tt - L:tt]
        sdv = torch.sqrt(look.var(0, unbiased=False) + 1e-5)
        ss = tt - H
        bx = x[ss - L:ss][None].to(dev); m.encoder.ids_shuffle = order.repeat(1, 1)
        f = m(bx, None, None, None)[0].cpu()
        eps_ref = (x[tt - H:tt] - f) / sdv
        assert (d['R'][j, H // P - 1].cpu() - eps_ref).abs().max() < 1e-4
        assert bool(d['obs'][j, H // P - 1].all())
        assert (d['yhat0'][j].cpu() - C['test']['Y'][tt - L]).abs().max() == 0
        assert (d['y'][j].cpu() - x[tt:tt + H]).abs().max() == 0
print('(2) cache == direct A0 forward (origins m = 0, 1, 2, 7); TEFL eps == residual of the forecast issued at t - H; targets = x[t:t+H]')
# (3) plug-ins
for arm in ('tefl', 'pos', 'ret', 'retfull'):
    torch.manual_seed(0)
    pl = Plugin(arm, H, M, P, dz=C['test']['Z'].shape[-1]).to(dev)
    p0 = pl(d)
    assert torch.equal(p0, d['yhat0']), arm
    loss = (p0 - d['y']).square().mean(); loss.backward()
    g = sum(p.grad.abs().sum().item() for p in pl.parameters() if p.grad is not None)
    with torch.no_grad():
        for p in pl.parameters():
            p.normal_(0, 0.05)
    pl.audit = 'zero'; pz = pl(d); pl.audit = None
    assert torch.equal(pz, d['yhat0']) and not torch.equal(pl(d), d['yhat0'])
    print(f'(3) {arm}: init == base, grad mass {g:.3e} > 0, zero audit == base'); assert g > 0
print('OK')
