"""PAT-307 lead-time diagnostic (eval only, no training): should the error fed back into rollout patch k be a lead-matched (k-patch-
ahead) error instead of the one-patch-ahead error that O / Q8 / the naive feedback use now?

A1 checkpoint, every (L, H) cell, all windows (stride 1). Lead of an error = number of patches between the forecast origin and the
forecast patch. The fed-back error always concerns the LAST lookback patch [t-P, t) (data scale, original variables):
  e1     = x[t-P, t) - A1's forecast of it from origin t-P (context x[:, :L-P], 1 patch)       lead 1 (O's latest slot, but with the
                                                                                               context's own RevIN instead of the window's)
  ek (k) = x[t-P, t) - the k-th patch of A1's rollout from origin t-k*P (context x[:, :L-k*P], k patches)        lead k
Each past forecast is A1 run on the truncated context alone (its own RevIN statistics, the fixed evaluation scan order): exactly the
forecast A1 would have issued at that origin. Leads are capped at T-1 (context >= one patch).
Rollout patch k (1-based, lead k from the origin t) gets  pred_k + beta * b, b = level (mean over the patch) or profile (per position):
  one1   b from e1,            ONE beta for all k            (= the current naive feedback with K=1)
  perk1  b from e1,            beta_k per rollout patch k     (isolates the per-k scale from the lead matching)
  match  b from e_min(k,T-1),  beta_k per rollout patch k     (lead-matched)
Every beta is least-squares fitted on VAL and applied to TEST. Reported: test MSE per variant and per rollout patch, and beta.
usage (repo root, via sbatch): python patchmem/lead_fb_eval.py --data E|W --seed 3047 [--grid 384:96,...] [--out path]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.mech_eval import DATA
from patchmem.naive_fb_eval import load

VARS = [f'{v}_{kind}' for v in ('one1', 'perk1', 'match') for kind in ('level', 'profile')]


@torch.no_grad()
def lead_errors(m, bx, n_out, P):
    """{lead k: (b, P, C) error of the last lookback patch, forecast from origin t-k*P}, k = 1..min(n_out, T-1)."""
    b, L, c = bx.shape
    T = L // P
    out = {}
    for k in range(1, min(n_out, T - 1) + 1):
        pk = m(bx[:, :L - k * P], k)['pred'][:, -P:]                                    # k-th rollout patch = [t-P, t)
        out[k] = bx[:, -P:] - pk
    return out


@torch.no_grad()
def run_cell(e, flag, L, H, fit=None, bs=64):
    """fit None: accumulate the least-squares terms of every beta (val). fit given: per-variant / per-patch squared error (test)."""
    m, dev = e.model, e.device
    P = m.P
    ds = e._ds(flag, L, H)
    ids = list(range(len(ds)))
    n_out = H // P
    T = L // P
    num = {(v, k): 0.0 for v in VARS for k in range(n_out)}                          # sum b*r per patch k
    den = {(v, k): 0.0 for v in VARS for k in range(n_out)}                          # sum b*b
    se = {v: np.zeros(n_out) for v in ['A1'] + VARS}
    nel = np.zeros(n_out)
    for s0 in range(0, len(ids), bs):
        bx, by = e._batch(ds, ids[s0:s0 + bs])
        bx, y = bx.to(dev), by[:, -H:].to(dev)
        pr = m(bx, n_out)['pred']
        b, c = pr.shape[0], pr.shape[-1]
        r = (y - pr).reshape(b, n_out, P, c)
        E = lead_errors(m, bx, n_out, P)
        for k in range(n_out):
            nel[k] += b * P * c
            se['A1'][k] += r[:, k].square().sum().item()
            for kind in ('level', 'profile'):
                f = (lambda z: z.mean(1, keepdim=True).expand_as(z)) if kind == 'level' else (lambda z: z)
                feats = {'one1': f(E[1]), 'perk1': f(E[1]), 'match': f(E[min(k + 1, T - 1)])}
                for v, bb in feats.items():
                    key = f'{v}_{kind}'
                    if fit is None:
                        num[(key, k)] += (bb * r[:, k]).sum().item(); den[(key, k)] += bb.square().sum().item()
                    else:
                        beta = fit[key] if v == 'one1' else fit[key][k]
                        se[key][k] += (r[:, k] - beta * bb).square().sum().item()
    if fit is None:                                                                     # one beta (one1) or beta_k (perk1, match)
        beta = {}
        for v in VARS:
            if v.startswith('one1'):
                beta[v] = sum(num[(v, k)] for k in range(n_out)) / max(sum(den[(v, k)] for k in range(n_out)), 1e-12)
            else:
                beta[v] = [num[(v, k)] / max(den[(v, k)], 1e-12) for k in range(n_out)]
        return beta
    return {v: dict(mse=float(se[v].sum() / nel.sum()), patch_mse=(se[v] / nel).tolist()) for v in se}, len(ids)


@torch.no_grad()
def check(e, flag, L, H, n=8):
    """Independent reference for lead_errors: the lead-k forecast of [t-P, t) is the official dataset window with seq_len L-k*P,
    pred_len k*P at index i - k*P (the split's border1 = border - seq_len shifts the index), forecast by A1, last patch."""
    m, P = e.model, e.model.P
    ds = e._ds(flag, L, H)
    ids = list(range(len(ds) - n, len(ds)))
    bx, _ = e._batch(ds, ids)
    bx = bx.to(e.device)
    E = lead_errors(m, bx, H // P, P)
    worst = 0.0
    for k, ek in E.items():
        dk = e._ds(flag, L - k * P, k * P)
        cx, cy = e._batch(dk, [i - k * P for i in ids])
        assert torch.equal(cx.to(e.device), bx[:, :L - k * P]) and torch.equal(cy[:, -P:].to(e.device), bx[:, -P:]), k
        ref = cy[:, -P:].to(e.device) - m(cx.to(e.device), k)['pred'][:, -P:]
        worst = max(worst, (ref - ek).abs().max().item())
    print(f'[lead check {flag} L={L} H={H}] leads 1..{max(E)} vs official shorter windows: max abs diff {worst:.2e}', flush=True)
    assert worst < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=['E', 'W']); ap.add_argument('--seed', type=int, default=3047)
    ap.add_argument('--grid', default=','.join(f'{L}:{H}' for L in (384, 720, 1440) for H in (96, 192, 336)))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    base, ck, _, _ = DATA[a.data]
    dA = {'E': 'R1h0s{}', 'W': 'R1Wh0s{}'}[a.data]
    e = load(base, ck, dA.format(a.seed), a.seed, {})
    out = a.out or f'patchmem/logs/lead/{a.data}_s{a.seed}'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    check(e, 'test', 384, 336)
    res = {}
    for g in a.grid.split(','):
        L, H = map(int, g.split(':'))
        beta = run_cell(e, 'val', L, H)
        r, nt = run_cell(e, 'test', L, H, fit=beta)
        res[f'{L}/{H}'] = dict(test=r, beta_val=beta, windows=nt)
        A = r['A1']['mse']
        print(f'[lead {a.data} s{a.seed}] L={L} H={H} windows {nt} | A1 {A:.4f} | ' +
              ' '.join(f'{v} {r[v]["mse"]:.4f} ({100 * (A - r[v]["mse"]) / A:+.1f}%)' for v in VARS), flush=True)
        for v in ['A1'] + VARS:
            print(f'   per-patch {v:14s} ' + ' '.join(f'{z:.4f}' for z in r[v]['patch_mse']), flush=True)
        print('   beta(val): ' + ' | '.join(f'{v} ' + (f'{beta[v]:.2f}' if isinstance(beta[v], float) else ' '.join(f'{z:.2f}' for z in beta[v]))
                                         for v in VARS), flush=True)
        json.dump(dict(data=a.data, seed=a.seed, grid=res), open(out + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
