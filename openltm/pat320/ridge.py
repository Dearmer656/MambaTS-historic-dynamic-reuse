"""PAT-320 arm b: EFC-style training-free per-series ridge on the series' own past errors (frozen joint Timer-XL).

For origin o of variable v: feature z_o = [1, E[o-96] binned, E[o-192] binned] (each one-step error token averaged over 8
bins of 12 steps -> 1 + 16 dims; E as in pat320/carry.py), target r_o = y[o:o+720] - rollout(o).
At test origin o the ridge is fitted on all pairs (z_o', r_o') of the same series whose target has matured (o' + 720 <= o)
and whose features use only non-training errors (o' - 192 >= t0); prediction Delta = z_o W. lambda is chosen per dataset
on the val stop origins (same set as carry.py) from a fixed grid. Cumulative sums make every origin's fit exact.
usage: python pat320/ridge.py --ckpt pat_logs/p320/joint_s2021.pth"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat320.joint import DATA, HS, L, P
from pat320.carry import HMAX, Frozen, build_cache, load_full, origins_val, sample, win

LAMS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, float("inf"))   # inf = no correction
NB = 8


@torch.no_grad()
def all_residuals(bb, x, o_lo, o_hi, bs=16384):
    """Residuals R [C, n, 720] fp16 (zero where the target leaves the series) for origins o_lo..o_hi of every variable,
    in large batches across variables (version 1 looped per variable: ECL took 2 h)."""
    T, C = x.shape; n = o_hi - o_lo + 1
    R = torch.empty(C, n, HMAX, dtype=torch.float16, device=x.device)
    vv, oo = torch.meshgrid(torch.arange(C, device=x.device), torch.arange(o_lo, o_hi + 1, device=x.device), indexing='ij')
    vv, oo = vv.reshape(-1), oo.reshape(-1); ar = torch.arange(HMAX, device=x.device)
    for i in range(0, len(oo), bs):
        v, o = vv[i:i + bs], oo[i:i + bs]
        base = bb.rollout(win(x, v, o - L, L).unsqueeze(-1))
        y = x[(o[:, None] + ar).clamp(max=T - 1), v[:, None]]
        R[v, o - o_lo] = ((y - base) * (ar[None] < (T - o)[:, None])).half()
    return R


def series_pairs(x, cache, t0, v, o_lo, o_hi, R_all):
    """Features Z [n, 17] and residuals R [n, 720] for origins o_lo..o_hi of variable v."""
    E, _ = cache
    o = torch.arange(o_lo, o_hi + 1, device=x.device)
    e1 = E[v, o - P - t0].float(); e2 = E[v, o - 2 * P - t0].float()
    Z = torch.cat([torch.ones(len(o), 1, device=x.device), e1.view(-1, NB, P // NB).mean(-1),
                   e2.view(-1, NB, P // NB).mean(-1)], 1)
    return o, Z, R_all[v].float()


def ridge_preds(o, Z, R, lam, query):
    """For query indices q into o: fit on pairs j with o[j] + 720 <= o[q] (cumulative sums), return Z[q] W."""
    d = Z.shape[1]
    A = torch.cumsum(Z[:, :, None] * Z[:, None, :], 0)                    # [n, d, d]
    Bm = torch.cumsum(Z[:, :, None] * R[:, None, :], 0)                   # [n, d, 720]
    out = torch.zeros(len(query), R.shape[1], device=Z.device)
    if lam == float("inf"):
        return out
    last = torch.searchsorted(o, o[query] - HMAX, right=True) - 1         # last matured pair index
    ok = last >= 0
    if ok.any():
        q, li = query[ok], last[ok]
        reg = lam * torch.eye(d, device=Z.device); reg[0, 0] = 1e-6       # intercept not shrunk
        W = torch.linalg.solve(A[li] + reg, Bm[li])
        out[ok] = torch.einsum('nd,ndh->nh', Z[q], W)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--data', default=','.join(DATA))
    ap.add_argument('--stop_n', type=int, default=2000); ap.add_argument('--out', default=str(ROOT / 'pat_logs' / 'p320'))
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bb = Frozen(a.ckpt, dev); res = dict(ckpt=a.ckpt, lams=LAMS, test={}, lam={})
    for d in a.data.split(','):
        t1 = time.time()
        bb.set(d); xn, b = load_full(d); x = torch.tensor(xn, device=dev); T, C = x.shape
        cache = build_cache(bb, x, b[0]); o_lo0 = b[0] + 2 * P; R_all = all_residuals(bb, x, o_lo0, T - HS[0])
        o_lo = b[0] + 2 * P                                                # first origin with two non-training error tokens
        (_, _), (slo, shi) = origins_val(b, 14)
        v_s, o_s = sample(np.random.default_rng(0), C, slo, shi, a.stop_n)   # same stop set as carry.py
        stop_sse = {lam: 0.0 for lam in LAMS}; base_sse = 0.0
        se = {lam: {H: 0.0 for H in HS} for lam in LAMS}; se0 = {H: 0.0 for H in HS}; cnt = {H: 0 for H in HS}
        for v in range(C):
            o, Z, R = series_pairs(x, cache, b[0], v, o_lo, T - HS[0], R_all)
            qs = torch.tensor(o_s[v_s == v] - o_lo, device=dev)
            qt = torch.arange(b[1] - o_lo, len(o), device=dev)               # official test origins
            for lam in LAMS:
                if len(qs):
                    stop_sse[lam] += (R[qs] - ridge_preds(o, Z, R, lam, qs)).square().mean(1).sum().item()
                pt = ridge_preds(o, Z, R, lam, qt)
                for H in HS:
                    m = o[qt] <= T - H
                    se[lam][H] += (R[qt][m, :H] - pt[m, :H]).square().sum().item()
            if len(qs):
                base_sse += R[qs].square().mean(1).sum().item()
            for H in HS:
                m = o[qt] <= T - H
                se0[H] += R[qt][m, :H].square().sum().item(); cnt[H] += int(m.sum()) * H
        lam = min(LAMS, key=lambda l: stop_sse[l]); res['lam'][d] = lam
        res['test'][d] = {H: dict(mse=se[lam][H] / cnt[H], base_mse=se0[H] / cnt[H]) for H in HS}
        rr = res['test'][d]
        print(f'[ridge] {d} | lam {lam} (stop {stop_sse[lam] / len(o_s):.4f} vs base {base_sse / len(o_s):.4f}) | '
              + ' | '.join(f'H{H} {rr[H]["mse"]:.4f} (base {rr[H]["base_mse"]:.4f})' for H in HS)
              + f' | avg {np.mean([rr[H]["mse"] for H in HS]):.4f} vs {np.mean([rr[H]["base_mse"] for H in HS]):.4f}'
              + f' | {time.time() - t1:.0f}s', flush=True)
        del cache, R_all
    json.dump(res, open(os.path.join(a.out, f'ridge_{Path(a.ckpt).stem}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
