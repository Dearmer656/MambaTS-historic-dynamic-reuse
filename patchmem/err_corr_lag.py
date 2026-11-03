"""PAT-315: how are iTransformer forecast errors at different target positions correlated?  e_o[h] = y_(o+h) - f_o[h].
Frozen backbone, Traffic, train / val / test origins separately (descriptive only, nothing is selected on test).  Per channel and lead the errors are z-scored on
the split; correlations are averaged over channels (mean of per-channel correlations).
  (1) same lead, target gap d:      A_h(d)   = corr(e_o[h], e_(o+d)[h]),                 d = 0 .. DMAX
  (2) same origin, lead gap i - j:  B[i, j]  = corr(e_o[i], e_o[j])                        (targets o+i, o+j)
  (3) same target, lead gap i - j:  Cm[i, j] = corr(e_(tau-i)[i], e_(tau-j)[j])             (target tau, origins tau-i, tau-j)
Usage: ITR_L=720 ITR_H=192 python -m patchmem.err_corr_lag --seed 3047
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_err_value as ev
from patchmem.eval_err_value import H, L
from patchmem.eval_itransformer_pure_icl import load_base

DMAX = 504                                                                        # 3 weeks


def zs(x):                                                                        # z-score over dim 0
    return (x - x.mean(0)) / x.std(0).clamp_min(1e-8)


def stats(E, dev):
    """E (n H C) cpu float32 for consecutive origins."""
    n, Hh, C = E.shape
    A = torch.zeros(Hh, DMAX + 1, dtype=torch.float64)
    nf = 1 << math.ceil(math.log2(2 * n))
    for h in range(Hh):                                                           # (1) per-lead ACF via FFT
        x = zs(E[:, h].to(dev).double())                                          # n C
        F = torch.fft.rfft(x, n=nf, dim=0)
        ac = torch.fft.irfft(F * F.conj(), n=nf, dim=0)[:DMAX + 1]                 # sum_o x_o x_(o+d)
        A[h] = (ac / (n - torch.arange(DMAX + 1, device=dev)[:, None])).mean(1).cpu()
    B = torch.zeros(Hh, Hh, dtype=torch.float64, device=dev)
    Cm = torch.zeros(Hh, Hh, dtype=torch.float64, device=dev)
    ts = torch.arange(Hh - 1, n)                                                  # targets tau = o0 + t, all leads observed
    for c0 in range(0, C, 64):
        Z = zs(E[:, :, c0:c0 + 64].to(dev).double())                              # n H c
        B += torch.einsum('nic,njc->ij', Z, Z) / n
        S = torch.stack([Z[ts - h, h] for h in range(Hh)], 1)                     # (n-H+1) H c : S[t, h] = e_(t-h)[h]
        S = zs(S)
        Cm += torch.einsum('tic,tjc->ij', S, S) / len(ts)
    return A.numpy(), (B / C).cpu().numpy(), (Cm / C).cpu().numpy()


def by_gap(M):
    Hh = M.shape[0]
    return np.array([np.mean(np.diagonal(M, k)) for k in range(Hh)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--out_dir', default='patchmem/logs/server/err_corr')
    a = p.parse_args()
    exp, _, _ = load_base('T', a.seed)
    dev = exp.device
    X, M, n_tr, n_va = ev.load_series(exp, 'T')
    o0 = L
    E = ev.stream_errors(exp, X, M, o0, 64)                                       # origins o0 .. len - H
    o = torch.arange(o0, o0 + len(E))
    out = {}
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    for name, m in (('train', o <= n_tr - H), ('val', (o >= n_tr) & (o <= n_tr + n_va - H)), ('test', o >= n_tr + n_va)):
        A, B, Cm = stats(E[m], dev)
        np.savez_compressed(od / f'T_H{H}_L{L}_s{a.seed}_{name}.npz', A=A, B=B, Cm=Cm)
        out[name] = dict(n=int(m.sum()))
        for h in (0, 23, 95, 191):
            out[name][f'A_h{h}'] = {f'd{d}': float(A[h, d]) for d in (1, 2, 6, 12, 24, 48, 72, 168, 336, 504)}
        bg, cg = by_gap(B), by_gap(Cm)
        out[name]['B_gap'] = {f'g{g}': float(bg[g]) for g in (1, 2, 6, 12, 24, 48, 96, 168, 191)}
        out[name]['C_gap'] = {f'g{g}': float(cg[g]) for g in (1, 2, 6, 12, 24, 48, 96, 168, 191)}
        print(f'[{name}] n {out[name]["n"]}', flush=True)
        for h in (0, 23, 95, 191):
            print(f'  A h{h:3d}: ' + ' '.join(f'{k} {v:+.3f}' for k, v in out[name][f'A_h{h}'].items()), flush=True)
        print('  B same origin, lead gap: ' + ' '.join(f'{k} {v:+.3f}' for k, v in out[name]['B_gap'].items()), flush=True)
        print('  C same target, lead gap: ' + ' '.join(f'{k} {v:+.3f}' for k, v in out[name]['C_gap'].items()), flush=True)
    (od / f'T_H{H}_L{L}_s{a.seed}.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
