"""PAT-315: value of the backbone's own errors, post hoc on the frozen standalone iTransformer (L96, H192), vs TEFL.

Stream errors E[o] = y[o:o+H] - f(o) for every origin o (f = standalone base, f reads y[o-L:o]).  At origin t the
observed error information is everything whose target time is <= t-1.  Feature blocks per (t, channel):
  eps   E[t-H]                      (H)  the matured vector issued one horizon ago -- TEFL's input
  u     E[t-1-j, lead 0], j < LU    (LU) one-step-ahead error history (freshest per target time)
  D     E[t-1-l, lead l], l < H     (H)  the newest observation diagonal (all forecasts of y[t-1])
Linear predictor of e_t = E[t] (H outputs), shared across channels, bias included; arms = subsets of the blocks
(eps, u, D, eps+u, eps+D, all).  Ridge W = (X'X + lam s I)^-1 X'Y, s = mean diag(X'X), lam in LAMS chosen per arm on
val MSE; fitted on train origins (targets inside train), "refit" = same lam refitted on train + val.
Everything is computed from second moments per split.  Reported: test MSE, gain vs base, vs TEFL (official
tefl_metrics.json of TEFL_paper_L96_E192, same test windows).  Criterion (fixed): arm < TEFL on 3/3 seeds per dataset.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
np.Inf = np.inf
from data_provider.data_factory import data_provider

L, H, LU = 96, 192, 96
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
TEFL_NAME = {'E': 'ETTh2', 'M1': 'ETTm1', 'M2': 'ETTm2', 'W': 'weather', 'T': 'traffic'}
BLOCKS = {'eps': (0, H), 'u': (H, H + LU), 'D': (H + LU, 2 * H + LU)}
ARMS = {'eps': ('eps',), 'u': ('u',), 'D': ('D',), 'eps+u': ('eps', 'u'), 'eps+D': ('eps', 'D'),
        'all': ('eps', 'u', 'D')}
NF = 2 * H + LU + 1                                                       # + bias


def borders(code, T):
    if code in ('E',):
        return 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
    if code in ('M1', 'M2'):
        return 12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4
    n_tr, n_te = int(T * 0.7), int(T * 0.2)
    return n_tr, T - n_tr - n_te, n_te


def load_series(exp, code):
    import pandas as pd
    args = exp.args
    T = len(pd.read_csv(Path(args.root_path) / args.data_path, usecols=[0]))
    n_tr, n_va, n_te = borders(code, T)
    old = args.seq_len
    args.seq_len = n_tr + n_va
    try:
        ds, _ = data_provider(args, 'test')
    finally:
        args.seq_len = old
    X = torch.as_tensor(np.asarray(ds.data_x), dtype=torch.float32)
    M = torch.as_tensor(np.asarray(ds.data_stamp), dtype=torch.float32)
    assert len(X) == n_tr + n_va + n_te and len(ds) == n_te - H + 1, (len(X), n_tr, n_va, n_te, len(ds))
    return X, M, n_tr, n_va


@torch.no_grad()
def stream_errors(exp, X, M, o0, bs):
    """E (n H C) for origins o0 .. len(X) - H (index o - o0)."""
    origins = torch.arange(o0, len(X) - H + 1)
    E = torch.empty(len(origins), H, X.shape[1])
    off = torch.arange(-L, 0)
    for i in range(0, len(origins), bs):
        o = origins[i:i + bs]
        p = exp.model(X[o[:, None] + off].to(exp.device), M[o[:, None] + off].to(exp.device), None, None).float().cpu()
        E[i:i + len(o)] = X[o[:, None] + torch.arange(H)] - p
    return E


def feats(E, t, o0, dev):
    """X (B C NF) and Y (B C H) for origins t (all features observed by t - 1)."""
    k = t - o0
    eps = E[k - H]                                                                   # B H C
    j = torch.arange(LU)
    u = E[k[:, None] - 1 - j, 0]                                                     # B LU C
    l = torch.arange(H)
    D = E[k[:, None] - 1 - l, l]                                                     # B H C
    one = torch.ones_like(u[:, :1])
    Xf = torch.cat([eps, u, D, one], 1).permute(0, 2, 1).to(dev).double()
    return Xf, E[k].permute(0, 2, 1).to(dev).double()


def moments(E, ts, o0, dev, chunk):
    xx = torch.zeros(NF, NF, dtype=torch.float64, device=dev)
    xy = torch.zeros(NF, H, dtype=torch.float64, device=dev)
    yy = torch.zeros(H, dtype=torch.float64, device=dev)
    n = 0
    for i in range(0, len(ts), chunk):
        Xf, Y = feats(E, ts[i:i + chunk], o0, dev)
        Xf, Y = Xf.reshape(-1, NF), Y.reshape(-1, H)
        xx += Xf.T @ Xf
        xy += Xf.T @ Y
        yy += Y.square().sum(0)
        n += len(Y)
    return dict(xx=xx, xy=xy, yy=yy, n=n)


def cols(arm):
    c = [torch.arange(*BLOCKS[b]) for b in ARMS[arm]] + [torch.tensor([NF - 1])]
    return torch.cat(c)


def solve(m, c, lam):
    A = m['xx'][c][:, c]
    s = A.diagonal()[:-1].mean()
    reg = lam * s * torch.eye(len(c), dtype=A.dtype, device=A.device)
    reg[-1, -1] = 0                                                                  # no penalty on the bias
    return torch.linalg.solve(A + reg, m['xy'][c])


def mse(m, c, W):
    return float((m['yy'].sum() - 2 * (W * m['xy'][c]).sum() + (W * (m['xx'][c][:, c] @ W)).sum()) / (m['n'] * H))


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    hits = glob.glob(f'results/long_term_forecast_{TEFL_NAME[code]}_96_192_TEFL_iTransformer_*'
                     f'TEFL_paper_L96_E192_s{seed}_0/tefl_metrics.json')
    assert len(hits) == 1, hits
    tefl = json.loads(Path(hits[0]).read_text())
    X, M, n_tr, n_va = load_series(exp, code)
    m_tr = args.m_tr if args.m_tr else n_tr
    assert m_tr >= 2 * H + 1, 'need eps / D history before the first fit origin'
    o0 = max(L, n_tr - m_tr)
    E = stream_errors(exp, X, M, o0, args.bs)
    t_tr = torch.arange(o0 + H, n_tr - H + 1)
    t_va = torch.arange(n_tr, n_tr + n_va - H + 1)
    t_te = torch.arange(n_tr + n_va, len(X) - H + 1)
    mo = {k: moments(E, ts, o0, dev, args.chunk) for k, ts in (('train', t_tr), ('val', t_va), ('test', t_te))}
    base = float(mo['test']['yy'].sum()) / (mo['test']['n'] * H)
    if abs(base - official) > 1e-4:
        raise AssertionError(f'base {base} official {official}')
    trva = {k: mo['train'][k] + mo['val'][k] for k in ('xx', 'xy', 'yy', 'n')}
    res = {}
    for arm in ARMS:
        c = cols(arm).to(dev)
        val = {lam: mse(mo['val'], c, solve(mo['train'], c, lam)) for lam in LAMS}
        lam = min(val, key=val.get)
        W = solve(mo['train'], c, lam)
        Wr = solve(trva, c, lam)
        r = dict(lam=lam, val_mse=val[lam], val_grid={str(k): v for k, v in val.items()},
                 test_mse=mse(mo['test'], c, W), refit_test_mse=mse(mo['test'], c, Wr))
        for k in ('test_mse', 'refit_test_mse'):
            r[k.replace('mse', 'gain')] = 100 * (base - r[k]) / base
            r[k.replace('mse', 'vs_tefl')] = 100 * (tefl['mse'] - r[k]) / tefl['mse']
        res[arm] = r
        print(f'[test] {arm:6s} lam {lam:g} mse {r["test_mse"]:.5f} gain {r["test_gain"]:+.2f}% vs TEFL '
              f'{r["test_vs_tefl"]:+.2f}% | refit {r["refit_test_mse"]:.5f} vs TEFL {r["refit_test_vs_tefl"]:+.2f}%',
              flush=True)
    out = dict(dataset=code, seed=seed, checkpoint=str(checkpoint), base=base, official=official,
               tefl_mse=tefl['mse'], tefl_gain_vs_base=100 * (base - tefl['mse']) / base, n_train_origins=len(t_tr),
               n_val=len(t_va), n_test=len(t_te), lams=LAMS, arms=res)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f'{code}_errval_s{seed}.json'
    p.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {p} base {base:.5f} TEFL {tefl["mse"]:.5f}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, choices=TEFL_NAME)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--m_tr', type=int, default=None, help='train origins kept before n_tr (default: all)')
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--out_dir', default='patchmem/logs/server/err_value')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
