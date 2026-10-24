"""PAT-307 task 6 baseline: EFC-CET (Error Feedback Calibration, "Look Back, Correct Forward", arXiv 2610.22872, Sec. CED / CET),
re-implemented on our frozen A1 (plain R1) forecasts. Training-free: closed-form ridge, no gradient (EFC's TRL GRU is NOT included).

Origin t: input x[t-L:t], forecast yhat_t(h) of y[t+h-1], h = 1..H (lead h); its error E_t[h] = y[t+h-1] - yhat_t(h).
  CED  D_t[h] = E_{t-h}[h], h = 1..H: the errors of the H earlier origins on the last observed value y[t-1] (available at t).
  CET  W_t = (lam I + sum_{i in M_t} D_i^T D_i)^-1 sum_{i in M_t} D_i^T E_i, M_t = {matured origins i: i + H <= t, CED of i complete},
       one H x H map shared by the variables (each variable's (D_i, E_i) is one row), statistics accumulated along the stream.
  alpha_t = clip_[0,1] ( sum_{i in V_t} <D_i W_v, E_i> / (sum ||D_i W_v||^2 + 1e-8) ), V_t = the last NV matured origins, W_v fitted on
       the matured origins before V_t (out of sample); EMA abar_t = RHO abar_{t-1} + (1 - RHO) alpha_t, abar = 0 until W_v exists.
  corrected forecast: yhat_t + abar_t D_t W_t   (no correction while M_t has fewer than NV + 1 origins or D_t is incomplete).
Choices the paper leaves open (fixed here before any result, recorded in the log header): W shared across variables; NV = H;
abar starts at 0; errors in RevIN-free data units (the dataset's standardised scale, as the official MSE).
Row feature (the 2x2 vs our window-local ICL; same CET / alpha machinery, only the feature changes): F_t = E_{t-H}[1..H], the full
error profile of the origin one horizon earlier (all its targets y[t-H..t-1] observed at t; lag H as ICL-0 with K = H). Streams
cold_row / warm_row / warm_row_a1.
Streams: 'cold' starts at the first official test origin (deployment = test start, EFC's setting); 'warm' starts WARM steps earlier
(errors of the preceding val period already in the statistics, evaluated on the same official test origins). Also reported: alpha = 1
(no calibration, warm) and the plain A1 forecast. MSE over the official test windows of each (L, H), A1 sanity-checked against exp_r1.
usage: python patchmem/efc_cet.py --data E|W [--seeds 3047,3048,3049] [--L 384,720,1440] [--H 96,192,336] [--out DIR] [--smoke]"""
import argparse, json, os, sys, time
from collections import deque
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
P, HMAX, NP = 48, 336, 7
LAM, RHO, WARM = 1.0, 0.95, 2880                                                       # EFC lambda / rho; warm prefix = ETTh2's val length
A1 = {'E': 'R1h0s{s}', 'W': 'R1Wh0s{s}'}


@torch.no_grad()
def rollout_errors(m, X, L, n0, bs=128):
    """Errors (N, HMAX, C) of the A1 7-patch rollout at origins t = L .. L+N-1 (N = n0); NaN where the target is beyond X."""
    Xt = torch.as_tensor(X)
    T = len(X)
    out = []
    for i0 in range(0, n0, bs):
        ts = range(L + i0, L + min(i0 + bs, n0))
        bx = torch.stack([Xt[t - L:t] for t in ts]).to(dev)
        pred = m(bx, NP)['pred'].float()
        y = torch.full_like(pred, float('nan'))
        for j, t in enumerate(ts):
            k = min(HMAX, T - t)
            y[j, :k] = Xt[t:t + k].to(dev)
        out.append(y - pred)
    return torch.cat(out)


def ced(E, H):
    """D (N, H, C): D[t, h-1] = E[t-h, h-1]; valid (N,) iff all H earlier origins exist."""
    N, _, C = E.shape
    D = torch.zeros(N, H, C, device=E.device)
    for h in range(1, H + 1):
        D[h:, h - 1] = E[:N - h, h - 1]
    valid = torch.arange(N, device=E.device) >= H
    return D, valid


def row_feat(E, H):
    """F (N, H, C): F[t] = E[t-H, :H]; valid (N,) iff origin t-H exists."""
    N, _, C = E.shape
    F = torch.zeros(N, H, C, device=E.device)
    F[H:] = E[:N - H, :H]
    return F, torch.arange(N, device=E.device) >= H


@torch.no_grad()
def stream(E, H, first, nv, calibrate=True, feat='ced'):
    """Run CET along origins 0..N-1; returns the correction (N, H, C) (0 where inactive) and the active mask. Only origins >= first
    are corrected (and reported); statistics use every matured origin with a complete CED."""
    N, _, C = E.shape
    Eh = E[:, :H]
    assert not torch.isnan(Eh).any(), 'stream input must have full targets (NaN would poison S / Tm)'
    D, valid = (ced if feat == "ced" else row_feat)(E, H)
    rows_D = D.permute(0, 2, 1)                                                          # (N, C, H): one row per variable
    rows_E = Eh.permute(0, 2, 1)
    eye = LAM * torch.eye(H, device=E.device, dtype=torch.float64)
    S = torch.zeros(H, H, device=E.device, dtype=torch.float64); Tm = torch.zeros_like(S)
    hist = []                                                                            # matured origins in order
    pS, pT = deque([S.clone()], maxlen=nv + 1), deque([Tm.clone()], maxlen=nv + 1)       # prefix sums over hist[:k], k = m-nv .. m
    corr = torch.zeros_like(Eh); active = torch.zeros(N, dtype=torch.bool, device=E.device)
    abar = 0.0
    for t in range(N):
        i = t - H                                                                        # origin i matures at t (i + H <= t)
        if i >= 0 and bool(valid[i]):
            d, e = rows_D[i].double(), rows_E[i].double()
            S = S + d.T @ d; Tm = Tm + d.T @ e
            hist.append(i); pS.append(S.clone()); pT.append(Tm.clone())
        m_ = len(hist)
        if m_ < nv + 1 or not bool(valid[t]):
            continue
        if calibrate:
            Wv = torch.linalg.solve(eye + pS[0], pT[0])                                  # prefix over hist[:m-nv]: before the window
            Vi = torch.as_tensor(hist[m_ - nv:], device=E.device)
            Q = rows_D[Vi].double() @ Wv                                                 # (nv, C, H)
            a = float(((Q * rows_E[Vi].double()).sum() / (Q.square().sum() + 1e-8)).clamp(0.0, 1.0))
            abar = RHO * abar + (1 - RHO) * a
        else:
            abar = 1.0
        if t >= first:
            W = torch.linalg.solve(eye + S, Tm)
            corr[t] = (abar * (rows_D[t].double() @ W)).float().T
            active[t] = True
    return corr, active


def main():
    from patchmem.diag_rollout import official_args
    from patchmem.resid_cov import CFG, split_x
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W'])
    ap.add_argument('--seeds', default='3047,3048,3049'); ap.add_argument('--L', default='384,720,1440')
    ap.add_argument('--H', default='96,192,336'); ap.add_argument('--out', default='patchmem/logs/server')
    ap.add_argument('--feats', default='ced,row', help='ced (EFC) and / or row (F_t = E_(t-H)) streams')
    ap.add_argument('--smoke', action='store_true', help='first seed, L 720, H 96, first 400 test origins')
    a_ = ap.parse_args()
    if a_.smoke:
        a_.seeds, a_.L, a_.H = a_.seeds.split(',')[0], '720', '96'
    cf = CFG[a_.data]
    for k in [k for k in os.environ if k.startswith(('R1_TEFL', 'R1_ICL', 'R1_OBS', 'R1_MEM', 'R1_PATHRES'))]:
        os.environ.pop(k)                                                                # plain R1 (A1)
    from patchmem.exp_r1 import Exp_R1
    os.makedirs(a_.out, exist_ok=True)
    res = {}
    import hashlib, subprocess
    md5 = hashlib.md5(open(__file__, 'rb').read()).hexdigest()[:10]
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    print(f'[version] efc_cet.py md5 {md5} | HEAD {head} | feats {a_.feats}', flush=True)
    print(f'EFC-CET on A1 | lam {LAM} rho {RHO} NV = H | warm prefix {WARM} | W shared by variables | abar starts at 0 | data units | features ced (EFC) + row (F_t = E_(t-H))', flush=True)
    for seed in [int(s) for s in a_.seeds.split(',')]:
        des = A1[a_.data].format(s=seed)
        ck = cf['r1ck'].replace(cf['r1des'], des)
        ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {des} --seed {seed}').split())
        ex = Exp_R1(ar)
        ex.model.load_state_dict(torch.load(f'checkpoints/{ck}/checkpoint.pth', map_location=dev))
        m = ex.model; m.eval(); ex._fix_eval_order()
        print(f'[ckpt] {ck} | {type(m).__name__}', flush=True)
        for L in [int(v) for v in a_.L.split(',')]:
            t0 = time.time()
            X = split_x(ar, L + WARM, 'test')                                            # WARM steps before the official test origins
            n0 = len(X) - L - min(int(h) for h in a_.H.split(',')) + 1
            E = rollout_errors(m, X, L, n0)
            for H in [int(v) for v in a_.H.split(',')]:
                first = WARM                                                             # index of the first official test origin
                last = len(X) - H - L                                                    # last official origin (inclusive)
                if a_.smoke:
                    last = first + 399
                sl = slice(first, last + 1)
                y_err = E[sl, :H]
                assert not torch.isnan(y_err).any(), 'official windows must have full targets'
                base = y_err.square().mean().item()
                row = dict(n=last - first + 1, base=base)
                for name, st, cal, ft in (('cold', first, True, 'ced'), ('warm', 0, True, 'ced'), ('warm_a1', 0, False, 'ced'),
                                          ('cold_row', first, True, 'row'), ('warm_row', 0, True, 'row'), ('warm_row_a1', 0, False, 'row')):
                    if ft not in a_.feats.split(','):
                        row[name] = row[name + '_active'] = float('nan'); continue
                    Es = E[st:last + 1]
                    if st > 0:                                                           # cold: the stream starts at the test start
                        c, act = stream(Es, H, 0, H, cal, ft)
                    else:
                        c, act = stream(Es, H, first, H, cal, ft)
                        c, act = c[first:], act[first:]
                    mse = (y_err - c).square().mean().item()
                    row[name] = mse; row[name + '_active'] = act.float().mean().item()
                res[f's{seed}|{L}|{H}'] = row
                print(f'  s{seed} L={L:4d} H={H:3d} | A1 {base:.4f} | CET cold {row["cold"]:.4f} ({(row["cold"] / base - 1) * 100:+.1f}%, '
                      f'active {row["cold_active"]:.2f}) | warm {row["warm"]:.4f} ({(row["warm"] / base - 1) * 100:+.1f}%, active '
                      f'{row["warm_active"]:.2f}) | warm alpha=1 {row["warm_a1"]:.4f} ({(row["warm_a1"] / base - 1) * 100:+.1f}%) '
                      f'| row: cold {row["cold_row"]:.4f} ({(row["cold_row"] / base - 1) * 100:+.1f}%) warm {row["warm_row"]:.4f} '
                      f'({(row["warm_row"] / base - 1) * 100:+.1f}%) a1 {row["warm_row_a1"]:.4f} | n {row["n"]} | {time.time() - t0:.0f}s', flush=True)
            json.dump(res, open(f'{a_.out}/efc_cet_{a_.data}{"_smoke" if a_.smoke else ""}.json', 'w'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
