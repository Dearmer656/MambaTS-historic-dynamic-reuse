"""PAT-315 design check for eval_sr: does the target-time shock history carry the linearly predictable error?

Same stream / ridge machinery as eval_err_value (iTransformer L96 H192, pooled channels, fit on train, lam on val,
test once).  Feature blocks per (t, channel), all observed by t - 1:
  sh    s_(t-1-j), j < 96        the shock history (s_tau = mean of the observation diagonal of tau)
  D     E[t-1-l, l], l < H        the newest diagonal (as eval_err_value)
  Dm    mean of D (= s_(t-1))     one number
  eps   E[t-H]                    the matured vector
Arms: sh, D, Dm, eps, sh+eps, D+eps.  If sh ~ D the shock carries the information (the carry model is the issue);
if sh << D the per-lead structure of the diagonal matters.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_err_value as ev
from patchmem.eval_err_value import H, L

LS = 96
BL = {'sh': (0, LS), 'D': (LS, LS + H), 'Dm': (LS + H, LS + H + 1), 'eps': (LS + H + 1, LS + 2 * H + 1)}
ARMS = {'sh': ('sh',), 'D': ('D',), 'Dm': ('Dm',), 'eps': ('eps',), 'sh+eps': ('sh', 'eps'), 'D+eps': ('D', 'eps')}
NF = LS + 2 * H + 2


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    X, M, n_tr, n_va = ev.load_series(exp, code)
    o0 = max(L, n_tr - args.m_tr) if args.m_tr else L
    E = ev.stream_errors(exp, X, M, o0, args.bs)                                     # n H C
    n = E.shape[0]
    ssum = torch.zeros(n + H - 1, E.shape[2])
    cnt = torch.zeros(n + H - 1)
    for l in range(H):
        ssum[l:l + n] += E[:, l]
        cnt[l:l + n] += 1
    shock = ssum / cnt[:, None]                                                      # index tau - o0

    def feats(t):
        k = t - o0
        j = torch.arange(LS)
        sh = shock[k[:, None] - 1 - j]                                               # B LS C
        l = torch.arange(H)
        D = E[k[:, None] - 1 - l, l]                                                 # B H C
        Dm = D.mean(1, keepdim=True)
        eps = E[k - H]
        one = torch.ones_like(Dm)
        Xf = torch.cat([sh, D, Dm, eps, one], 1).permute(0, 2, 1).to(dev).double()
        return Xf, E[k].permute(0, 2, 1).to(dev).double()

    def moments(ts):
        xx = torch.zeros(NF, NF, dtype=torch.float64, device=dev)
        xy = torch.zeros(NF, H, dtype=torch.float64, device=dev)
        yy = torch.zeros(H, dtype=torch.float64, device=dev)
        m = 0
        for i in range(0, len(ts), args.chunk):
            Xf, Y = feats(ts[i:i + args.chunk])
            Xf, Y = Xf.reshape(-1, NF), Y.reshape(-1, H)
            xx += Xf.T @ Xf
            xy += Xf.T @ Y
            yy += Y.square().sum(0)
            m += len(Y)
        return dict(xx=xx, xy=xy, yy=yy, n=m)

    first = o0 + max(H, LS + H)                                                      # full shock history
    t_tr = torch.arange(first, n_tr - H + 1)
    t_va = torch.arange(n_tr, n_tr + n_va - H + 1)
    t_te = torch.arange(n_tr + n_va, len(X) - H + 1)
    mo = {k: moments(ts) for k, ts in (('train', t_tr), ('val', t_va), ('test', t_te))}
    base = float(mo['test']['yy'].sum()) / (mo['test']['n'] * H)
    assert abs(base - official) < 1e-4, (base, official)
    res = {}
    for arm, bl in ARMS.items():
        c = torch.cat([torch.arange(*BL[b]) for b in bl] + [torch.tensor([NF - 1])]).to(dev)
        val = {lam: ev.mse(mo['val'], c, ev.solve(mo['train'], c, lam)) for lam in ev.LAMS}
        lam = min(val, key=val.get)
        te = ev.mse(mo['test'], c, ev.solve(mo['train'], c, lam))
        res[arm] = dict(lam=lam, test_mse=te, gain=100 * (base - te) / base)
        print(f'[test] {arm:7s} lam {lam:g} gain {res[arm]["gain"]:+.2f}%', flush=True)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f'{code}_shockridge_s{seed}.json'
    p.write_text(json.dumps(dict(dataset=code, seed=seed, H=H, base=base, arms=res), indent=2) + '\n')
    print(f'[done] {p}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, choices=('E', 'M1', 'M2', 'W', 'T', 'EL'))
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--m_tr', type=int, default=None)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--out_dir', default='patchmem/logs/server/err_sr_diag')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
