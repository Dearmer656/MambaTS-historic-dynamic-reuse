"""PAT-315: is the periodic forecast error an amplitude miscalibration?  One shared scalar per dataset (and seed).

Frozen standalone iTransformer (L96, H192, existing checkpoints, no retraining).  For every origin o and channel c:
  raw       y^' = alpha y^
  centered  y^' = mu + alpha (y^ - mu),   mu = mean of the input window y[o-L:o, c]
alpha in 0.700 .. 1.300 step 0.025 (alpha = 1 = backbone).  MSE(alpha) over all origins x leads x channels is
quadratic in alpha, so each split keeps three sums: S_aa = sum (y - a)^2, S_ab = sum (y - a) b, S_bb = sum b^2 with
(a, b) = (0, y^) for raw and (mu, y^ - mu) for centered; MSE(alpha) = (S_aa - 2 alpha S_ab + alpha^2 S_bb) / N.
alpha chosen on val MSE only (val origins o in [n_tr, n_tr + n_va - H], targets inside val), test once (standard test
windows, base must equal the official test MSE).  Also reported (not used for selection): the val-optimal continuous
alpha* = S_ab / S_bb.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_err_value as ev
from patchmem.eval_err_value import H, L

ALPHAS = np.round(np.arange(0.7, 1.3 + 1e-9, 0.025), 3)
KINDS = ('raw', 'centered')


@torch.no_grad()
def sums(exp, X, M, origins, bs):
    """Per kind: (S_aa, S_ab, S_bb) in float64 and the element count."""
    s = {k: np.zeros(3) for k in KINDS}
    n = 0
    off, lead = torch.arange(-L, 0), torch.arange(H)
    for i in range(0, len(origins), bs):
        o = origins[i:i + bs]
        xin = X[o[:, None] + off]                                                    # B L C
        p = exp.model(xin.to(exp.device), M[o[:, None] + off].to(exp.device), None, None).double()
        y = X[o[:, None] + lead].to(exp.device).double()                             # B H C
        mu = xin.to(exp.device).double().mean(1, keepdim=True)                       # B 1 C
        for k, (a, b) in {'raw': (torch.zeros_like(y), p), 'centered': (mu.expand_as(y), p - mu)}.items():
            r = y - a
            s[k] += [float(r.square().sum()), float((r * b).sum()), float(b.square().sum())]
        n += y.numel()
    return s, n


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    X, M, n_tr, n_va = ev.load_series(exp, code)
    split = {'val': torch.arange(n_tr, n_tr + n_va - H + 1), 'test': torch.arange(n_tr + n_va, len(X) - H + 1)}
    st = {z: sums(exp, X, M, o, args.bs) for z, o in split.items()}
    mse = lambda z, k, al: (st[z][0][k][0] - 2 * al * st[z][0][k][1] + al ** 2 * st[z][0][k][2]) / st[z][1]
    base = mse('test', 'raw', 1.0)
    assert abs(base - official) < 1e-4, (base, official)
    out = dict(dataset=code, seed=seed, H=H, L=L, checkpoint=str(checkpoint), base_test=base, base_val=mse('val', 'raw', 1.0),
               alphas=ALPHAS.tolist(), n_val=len(split['val']), n_test=len(split['test']))
    for k in KINDS:
        curve = [mse('val', k, a) for a in ALPHAS]
        ab = ALPHAS[int(np.argmin(curve))]
        te = mse('test', k, ab)
        out[k] = dict(val_curve=curve, alpha=float(ab), test_mse=te,                    # test only at the val-chosen alpha
                      gain=100 * (base - te) / base, val_gain=100 * (out['base_val'] - min(curve)) / out['base_val'],
                      alpha_star_val=st['val'][0][k][1] / max(st['val'][0][k][2], 1e-30))
        print(f'[{code} s{seed}] {k:8s} alpha {ab:.3f} (val* {out[k]["alpha_star_val"]:.3f}) test {te:.5f} '
              f'gain {out[k]["gain"]:+.3f}% (val gain {out[k]["val_gain"]:+.3f}%)', flush=True)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f'{code}_H{H}_s{seed}.json').write_text(json.dumps(out, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='W,T,E,M1')
    p.add_argument('--seeds', default='3047,2024,2025')
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--out_dir', default='patchmem/logs/server/amp_scale')
    a = p.parse_args()
    for code in a.data.split(','):
        for s in a.seeds.split(','):
            run(code, int(s), a)


if __name__ == '__main__':
    main()
