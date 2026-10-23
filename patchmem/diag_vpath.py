"""PAT-307 checks on the TRAINED ETTh2 checkpoints (seed 3047): original ErrPath (vpath, C*P), ErrPath-shared (vshare, P) and O.
1 reload: the test MSE at L=1440/H=336 reproduces the training log (same checkpoint, same eval scan order).
2 window alignment: every test window's x is data[s:s+L] and its target is data[s+L:s+L+H] (no target row inside the lookback).
3 no cross-sample leakage: a window forecast alone (batch of 1) == the same window inside a batch of 64 overlapping windows
  (sliding test windows: a neighbour's lookback contains this window's target).
4 normalisation: the dataset scaler is fitted on the train split only (its mean == the train rows' mean of the raw csv).
(The forecast is a function of the lookback x only: R1Obs.forward takes no target, so 2 + 3 + 4 cover the routes a target could leak by.)
5 path usage: per Householder on the stored errors, beta = 2 sigmoid(W_bv h), |cos(v, w)| and the relative change |Hv - v| / |v|
  (per variable for vshare), and the end-to-end change |V_path - V| / |V| of the values read at rollout patch k.
usage: python patchmem/diag_vpath.py"""
import copy, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from patchmem.diag_rollout import official_args, ETTH2

CK = 'checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_{}_0/checkpoint.pth'
ARMS = {'ErrPath': ('R1ObK0.5Q8PVh0s3047', dict(R1_OBS_VPATH='1'), 0.379396),
        'ErrPath-shared': ('R1ObK0.5Q8PVsh0s3047', dict(R1_OBS_VPATH='1', R1_OBS_VSHARE='1'), 0.399338),
        'O': ('R1ObK0.5Q8h0s3047', {}, 0.387890)}
L, H, P = 1440, 336, 48
dev = 'cuda'
out = {}


def build(des, env):
    for k in ('R1_OBS_VPATH', 'R1_OBS_VSHARE', 'R1_OBS_PATH'):
        os.environ.pop(k, None)
    os.environ.update(R1_OBS='1', R1_MEM_QK_DIM='8', R1_MEM_SLOT_KEEP='0.5', **env)
    from patchmem.exp_r1 import Exp_R1
    a = official_args(ETTH2 + ['--des', des, '--seed', '3047'])
    ex = Exp_R1(a)
    ex.model.load_state_dict(torch.load(CK.format(des), map_location=dev))
    ex.model.eval(); ex._fix_eval_order()
    return ex


torch.manual_seed(0)
for name, (des, env, logged) in ARMS.items():
    ex = build(des, env); m = ex.model; r = {}
    with torch.no_grad():
        mse = ex.eval_grid('test', [(L, H)])[(L, H)]['mse']
        m.eval(); ex._fix_eval_order()
        r['1_reload'] = dict(mse=mse, logged=logged, diff=mse - logged)
        print(f'[{name}] 1 reload test {L}/{H} mse {mse:.6f} | log {logged:.6f} | diff {mse - logged:+.2e}', flush=True)
        ds = ex._ds('test', L, H)
        n = len(ds); ids = list(range(n))
        bad = 0
        for i in ids:
            s = i                                                                       # Dataset_ETT_hour: s_begin = index
            x, y = ds[i][0], ds[i][1][-H:]
            bad += int(not (np.array_equal(x, ds.data_x[s:s + L]) and np.array_equal(y, ds.data_y[s + L:s + L + H])))
        r['2_alignment_bad'] = bad
        print(f'[{name}] 2 window alignment: {bad} of {len(ids)} windows misaligned', flush=True)
        bx, by = ex._batch(ds, list(range(0, 64)))                                      # 64 consecutive (overlapping) windows
        bx = bx.to(dev)
        pb = m(bx, H // P)['pred']
        d3 = max((m(bx[i:i + 1], H // P)['pred'][0] - pb[i]).abs().max().item() for i in range(64))
        r['3_batch_vs_alone_maxabs'] = d3
        print(f'[{name}] 3 batch of 64 vs alone: max |diff| {d3:.2e} (pred scale {pb.abs().mean().item():.3f})', flush=True)
        import pandas as pd
        df = pd.read_csv('dataset/ETT-small/ETTh2.csv')
        tr = df[df.columns[1:]].values[:12 * 30 * 24]
        d4 = float(np.abs(ds.scaler.mean_ - tr.mean(0)).max())
        r['4_scaler_train_mean_maxabs'] = d4
        print(f'[{name}] 4 scaler mean vs train-split mean: max |diff| {d4:.2e}', flush=True)
        net = m.obs_net
        if net.vpath:
            rec = []
            tv = net.transport_v
            def hook(h, V):
                Vo = tv(h, V)
                z = h.detach().flatten(1)
                w = torch.nn.functional.normalize(net.Wwv(z), dim=-1)
                beta = 2 * torch.sigmoid(net.Wbv(z))[:, 0]
                if net.vshare:
                    cos = (torch.einsum('bmcp,bp->bmc', V, w) / V.norm(dim=-1).clamp_min(1e-12)).abs()
                    rel = (Vo - V).norm(dim=-1) / V.norm(dim=-1).clamp_min(1e-12)
                else:
                    Vf = V.flatten(2)
                    cos = (torch.einsum('bmd,bd->bm', Vf, w) / Vf.norm(dim=-1).clamp_min(1e-12)).abs()
                    rel = (Vo - V).flatten(2).norm(dim=-1) / Vf.norm(dim=-1).clamp_min(1e-12)
                rec.append((beta, cos.flatten(1).mean(1), rel.flatten(1).mean(1)))
                return Vo
            net.transport_v = hook
            seen = []
            f0 = net.forward
            net.forward = lambda h, K, V, mask=None: (seen.append(V.clone()), f0(h, K, V, mask))[1]
            pb2 = m(bx, H // P)['pred']
            net.transport_v = tv; net.forward = f0
            assert torch.equal(pb2, pb)
            net.audit = 'noH'; seen0 = []
            net.forward = lambda h, K, V, mask=None: (seen0.append(V.clone()), f0(h, K, V, mask))[1]
            m(bx, H // P); net.forward = f0; net.audit = None
            beta = torch.stack([t[0] for t in rec]); cos = torch.stack([t[1] for t in rec]); rel = torch.stack([t[2] for t in rec])
            assert len(seen) == len(seen0) == H // P, (len(seen), len(seen0))
            e2e = [((a - b).norm() / b.norm()).item() for a, b in zip(seen, seen0)]
            r['5_path'] = dict(n_householders=len(rec), beta_mean=beta.mean().item(), beta_min=beta.min().item(), beta_max=beta.max().item(),
                               abs_cos_mean=cos.mean().item(), rel_change_per_H=rel.mean().item(), end_to_end_rel_change_per_k=e2e)
            print(f'[{name}] 5 path: {len(rec)} transport_v calls (history + rollout) | beta mean {beta.mean():.3f} [{beta.min():.3f}, {beta.max():.3f}] | '
                  f'|cos(v,w)| {cos.mean():.4f} | |Hv-v|/|v| {rel.mean():.4f} | end-to-end |V_path-V|/|V| per rollout patch '
                  + ' '.join(f'{v:.3f}' for v in e2e), flush=True)
            print(f'[{name}]   random-direction reference |cos| ~ sqrt(2/(pi d)) = {np.sqrt(2 / (np.pi * net.Wwv.out_features)):.4f} '
                  f'(d = {net.Wwv.out_features})', flush=True)
    out[name] = r
    del ex, m; torch.cuda.empty_cache()
json.dump(out, open('patchmem/logs/server/diag_vpath_s3047.json', 'w'), indent=1)
print('DONE')
