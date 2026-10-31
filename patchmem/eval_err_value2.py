"""PAT-315 error value, stage 2: per-channel linear error feedback shrunk toward the pooled map.

Same stream, features and splits as eval_err_value (eps = E[t-H], u = one-step error history, D = newest diagonal).
Pooled W_pool = ridge on all channels (lam1 chosen on val, as stage 1).  Per channel c:
  W_c = argmin |Y_c - X_c W|^2 + lam2 s_c |W - W_pool|^2 = (A_c + lam2 s_c I)^-1 (b_c + lam2 s_c W_pool)
s_c = mean diag(A_c); lam2 in LAM2 chosen on val (lam2 -> inf = pooled).  Feature sets: eps+D, all.
Criterion (fixed): per-channel arm test MSE < TEFL on 3/3 seeds per dataset.
Horizon sweep: H from env ITR_H (see eval_err_value); secondary per-channel Delta(H) = pc_gain(all) - pc_gain(eps),
run on the small-C datasets only (E M1 M2 W; T/EL per-channel moments C x NF^2 do not fit at H720).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_err_value as ev
from patchmem.eval_err_value import H, NF

LAM2 = (0.01, 0.1, 1.0, 10.0, 100.0)
SETS = ('eps', 'eps+D', 'all')


def moments_pc(E, ts, o0, dev, chunk, C):
    xx = torch.zeros(C, NF, NF, dtype=torch.float64, device=dev)
    xy = torch.zeros(C, NF, H, dtype=torch.float64, device=dev)
    yy = torch.zeros(C, dtype=torch.float64, device=dev)
    n = 0
    for i in range(0, len(ts), chunk):
        Xf, Y = ev.feats(E, ts[i:i + chunk], o0, dev)                               # B C NF, B C H
        xx += torch.einsum('bci,bcj->cij', Xf, Xf)
        xy += torch.einsum('bci,bch->cih', Xf, Y)
        yy += Y.square().sum((0, 2))
        n += Xf.shape[0]
    return dict(xx=xx, xy=xy, yy=yy, n=n)                                             # n = origins per channel



def mse_pc(m, c, W):
    """W (C k H) or (k H)."""
    A, b = m['xx'][:, c][:, :, c], m['xy'][:, c]
    if W.dim() == 2:
        W = W.expand(A.shape[0], -1, -1)
    sse = m['yy'] - 2 * (W * b).sum((1, 2)) + (W * (A @ W)).sum((1, 2))
    return float(sse.sum() / (m['n'] * A.shape[0] * H))


def solve_pc(m, c, W0, lam2):
    A, b = m['xx'][:, c][:, :, c], m['xy'][:, c]
    s = A.diagonal(dim1=1, dim2=2)[:, :-1].mean(1).clamp_min(1e-12)
    reg = lam2 * s[:, None, None] * torch.eye(len(c), dtype=A.dtype, device=A.device)
    return torch.linalg.solve(A + reg, b + reg @ W0)


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    tefl = ev.load_tefl(code, seed)
    X, M, n_tr, n_va = ev.load_series(exp, code)
    m_tr = args.m_tr if args.m_tr else n_tr
    assert m_tr >= 2 * H + 1
    o0 = max(ev.L, n_tr - m_tr)
    E = ev.stream_errors(exp, X, M, o0, args.bs)
    C = X.shape[1]
    t_tr = torch.arange(o0 + H, n_tr - H + 1)
    t_va = torch.arange(n_tr, n_tr + n_va - H + 1)
    t_te = torch.arange(n_tr + n_va, len(X) - H + 1)
    mo = {k: moments_pc(E, ts, o0, dev, args.chunk, C) for k, ts in (('train', t_tr), ('val', t_va), ('test', t_te))}
    base = float(mo['test']['yy'].sum()) / (mo['test']['n'] * C * H)
    if abs(base - official) > 1e-4:
        raise AssertionError(f'base {base} official {official}')
    trp = {k: mo['train'][k].sum(0) for k in ('xx', 'xy')}
    trp['n'] = mo['train']['n'] * C
    res = {}
    for arm in SETS:
        c = ev.cols(arm).to(dev)
        val1 = {lam: mse_pc(mo['val'], c, ev.solve(trp, c, lam)) for lam in ev.LAMS}
        lam1 = min(val1, key=val1.get)
        W0 = ev.solve(trp, c, lam1)
        val2 = {lam2: mse_pc(mo['val'], c, solve_pc(mo['train'], c, W0, lam2)) for lam2 in LAM2}
        lam2 = min(val2, key=val2.get)
        pc = mse_pc(mo['test'], c, solve_pc(mo['train'], c, W0, lam2))
        po = mse_pc(mo['test'], c, W0)
        r = dict(lam1=lam1, lam2=lam2, val_pooled=val1[lam1], val_pc=val2[lam2], val2_grid={str(k): v for k, v in val2.items()},
                 pooled_test_mse=po, pc_test_mse=pc)
        for k in ('pooled', 'pc'):
            r[f'{k}_gain'] = 100 * (base - r[f'{k}_test_mse']) / base
            r[f'{k}_vs_tefl'] = 100 * (tefl['mse'] - r[f'{k}_test_mse']) / tefl['mse'] if tefl else None
        res[arm] = r
        print(f'[test] {arm:6s} pooled {po:.5f} gain {r["pooled_gain"]:+.2f}% | per-channel lam2 {lam2:g} '
              f'{pc:.5f} gain {r["pc_gain"]:+.2f}%', flush=True)
    out = dict(dataset=code, seed=seed, H=H, checkpoint=str(checkpoint), base=base, official=official,
               tefl_mse=tefl['mse'] if tefl else None, lam2_grid=LAM2, arms=res,
               delta_fresh_pc=res['all']['pc_gain'] - res['eps']['pc_gain'])
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    p = od / ev.out_name(code, seed, '2')
    p.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {p} base {base:.5f} pc Delta {out["delta_fresh_pc"]:+.2f}pp', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, choices=('E', 'M1', 'M2', 'W'))
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--m_tr', type=int, default=None)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--out_dir', default='patchmem/logs/server/err_value')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
