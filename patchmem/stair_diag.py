"""PAT-313 diagnostic: how much of the current forecast's error row is linearly predictable from the FRESH, partially observed errors
that the method ignores (the "staircase"), beyond the stale full row it uses? Training-free, TEFL checkpoints (fixed H), L1440 test.
Errors E[t, lead] = y - forecast of the (TEFL-corrected) rollout issued at step t (stride 1 step). At origin t the target row is
Y_t = E_t[:H]. Feature sets (all observed before t):
  row    E_{t-H}[:H]                                  the full row of the forecast issued H steps ago (what TEFL / hyb use)
  stair  E_{t-48k}[:48k], k = 1..K-1                  forecasts issued k patches ago: their first k patches are observed
  ced    E_{t-h}[h-1], h = 1..H                       every past forecast's error on the latest step t-1 (EFC's feature)
  row+stair, row+ced
Ridge pooled over variables (lam 1, as gain_src.oracle). Reported: in-sample removable share (upper bound) and the honest share
when fit on the first half of the test origins and scored on the second half.
usage: python patchmem/stair_diag.py --data E|W|T --H 192 [--seed 3047]"""
import argparse, os, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import patchmem.efc_cet as ec

P, L = 48, 1440
DES = {('E', 192): 'R1TfmsH192s{s}', ('W', 96): 'R1WTfmsH96s{s}', ('W', 192): 'R1WTfmsH192s{s}',
       ('T', 96): 'R1TTfmsH96s{s}', ('T', 192): 'R1TTfmsH192s{s}'}


def ridge_fit(F, Y, lam=1.0):
    S = torch.einsum('ncd,nce->de', F, F); T = torch.einsum('ncd,nce->de', F, Y)
    return torch.linalg.solve(lam * torch.eye(F.shape[-1], dtype=F.dtype, device=F.device) + S, T)


def share(F, Y, W):
    return float(1 - (Y - F @ W).square().sum() / Y.square().sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--H', type=int, default=192)
    ap.add_argument('--seed', type=int, default=3047)
    a_ = ap.parse_args()
    D, H, s = a_.data, a_.H, a_.seed
    K = H // P
    os.environ['R1_TEFL'] = 'ms'; os.environ['R1_HMAX'] = str(K)
    from patchmem.diag_rollout import official_args
    from patchmem.resid_cov import CFG, split_x
    from patchmem.exp_r1 import Exp_R1
    cf = CFG[D]
    des = DES[(D, H)].format(s=s)
    ck = cf['r1ck'].replace(cf['r1des'], des)
    ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {des} --seed {s}').split())
    ex = Exp_R1(ar)
    ex.model.load_state_dict(torch.load(f'checkpoints/{ck}/checkpoint.pth', map_location=ec.dev))
    ex.model.eval(); ex._fix_eval_order()
    X = split_x(ar, L, 'test')
    n0 = len(X) - L - H + 1
    with torch.no_grad():
        E = ec.rollout_errors(ex.model, X, L, n0)[:, :H].double().cpu()               # (N, H, C)
    base_mse = float(E.square().mean())
    N, _, C = E.shape
    t = torch.arange(H, N)                                                            # every feature observed
    Y = E[t].permute(0, 2, 1)                                                         # (n, C, H)
    row = E[t - H].permute(0, 2, 1)
    st = [E[t - P * k][:, :P * k] for k in range(1, K)]
    stair = torch.cat(st, 1).permute(0, 2, 1) if st else torch.zeros(len(t), C, 0, dtype=E.dtype)
    hh = torch.arange(1, H + 1)
    ced = E[t[:, None] - hh[None], hh[None] - 1].permute(0, 2, 1)                     # (n, C, H): E[t-h, h-1]
    sets = {'row': row, 'stair': stair, 'ced': ced, 'row+stair': torch.cat([row, stair], -1), 'row+ced': torch.cat([row, ced], -1)}
    n = len(t); h1, h2 = slice(0, n // 2), slice(n // 2, n)
    print(f'[stair] {D} H{H} s{s} {des} | test MSE of the TEFL rollout {base_mse:.4f} | origins {n} | K {K} | dims ' +
          ' '.join(f'{k} {v.shape[-1]}' for k, v in sets.items()), flush=True)
    for k, F in sets.items():
        if F.shape[-1] == 0:
            continue
        ins = share(F, Y, ridge_fit(F, Y))
        hon = share(F[h2], Y[h2], ridge_fit(F[h1], Y[h1]))
        print(f'[stair] {D} H{H} s{s} {k:10s} in-sample {ins:.3f} | fit half 1 -> half 2 {hon:+.3f}', flush=True)


if __name__ == '__main__':
    main()
