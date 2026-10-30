"""PAT-315: one-layer W1 ICL with every demo inside the history window, plus multi-step (stacked) in-case GD.

Same model as eval_raw_multistep (key = cos on E_{o-H} and on y[o-96:o], value E_s, kernel GD f_{k+1} = f_k +
eta sum_j a(., x_j)(e_j - f_k(x_j)), self / loo), but the demo pool is no longer only the same clock:
  phase   ages = multiples of 168 (the current 14 demos at H192)
  daily   ages = multiples of 24 (same hour of day)
  hourly  every age  (H .. history - H - L; ~2400 demos at H192)
Ages always >= H, so every demo value E_s is fully observed at t.  Logit = beta * (tau_e se + tau_y sy) with
tau = (1, 4) and beta in BETAS (the dense pools need a sharper softmax).
The query output after K steps is computed without the n x n x H demo update:
  f_K(q) = eta sum_{k<K} u_k V,   u_0 = a_q,   u_{k+1} = u_k (I - eta A_dd)
which equals the explicit recursion (tested).  beta, eta, K, s are selected per (pool, mode) on a fixed random train
subset (--select_cases); test once on a fixed random test subset (--test_cases, the same cases for every pool; the
hourly pool is too slow for all 2.9M test cases).  Gains are relative to the backbone on that subset.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_error_path import CELLS, H, L, all_cases, load_split
from patchmem.eval_raw_sim import S_GRID, mse_at

TAU_E, TAU_Y, YL = 1.0, 4.0, 96
KS = (1, 2, 3, 5, 10)
ETAS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
BETAS = (0.5, 1.0, 2.0, 4.0)
MODES = ('self', 'loo')
POOLS = {'phase': 168, 'daily': 24, 'hourly': 1}
BS = {'phase': 2048, 'daily': 512, 'hourly': 12}                 # hourly 12: two runs fit one 24 GB card


def pool_ages(pool, history):
    step, last = POOLS[pool], history - H - L
    first = ((H + step - 1) // step) * step
    return list(range(first, last + 1, step))


def cosm(a, b):
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-1, -2)


@torch.no_grad()
def sums(data, ages, device, bs, cands, max_cases=None, seed=0):
    """cands: set of (mode, beta, eta, K) -> [sum t^2, sum t.f, sum f^2, n] for f = f_K(query)."""
    E, X = data['E'], data['X']
    ag = torch.as_tensor(ages, device=device)
    k = torch.arange(YL, device=device)
    t_all, c_all = all_cases(data, max_cases, seed)
    acc = {c: torch.zeros(3, dtype=torch.float64, device=device) for c in cands}
    kmax = max(c[3] for c in cands)
    n = 0
    for i in range(0, len(t_all), bs):
        t, c = t_all[i:i + bs].to(device), c_all[i:i + bs].to(device)
        s = t[:, None] - ag[None]
        cc = c[:, None]
        q, target = E[t - H - L, :, c], E[t - L, :, c].double()
        K, V = E[s - H - L, :, cc], E[s - L, :, cc].double()
        yq = X[(t - 1)[:, None] - k, c[:, None]]
        ys = X[(s - 1)[..., None] - k, cc[..., None]]
        lq = (TAU_E * cosm(q[:, None], K) + TAU_Y * cosm(yq[:, None], ys))[:, 0].double()   # B n
        ldd = (TAU_E * cosm(K, K) + TAU_Y * cosm(ys, ys)).double()                         # B n n
        tt = target.square().sum()
        n += target.numel()
        for mode in MODES:
            dd = ldd.clone()
            if mode == 'loo':
                dd.diagonal(dim1=-2, dim2=-1).fill_(float('-inf'))
            for beta in sorted({c[1] for c in cands if c[0] == mode}):
                aq = torch.softmax(beta * lq, -1).float()
                add = torch.softmax(beta * dd, -1).float()    # fp32: the n x n matvecs dominate
                for eta in sorted({c[2] for c in cands if c[0] == mode and c[1] == beta}):
                    u, w = aq, torch.zeros_like(aq)
                    for step in range(1, kmax + 1):
                        w = w + eta * u
                        if (mode, beta, eta, step) in acc:
                            fq = (w.double()[..., None] * V).sum(1)
                            acc[mode, beta, eta, step] += torch.stack([tt, (target * fq).sum(), fq.square().sum()])
                        u = u - eta * (u[:, None] @ add)[:, 0]
    return {c: torch.cat([a.cpu(), torch.tensor([float(n)], dtype=torch.float64)]) for c, a in acc.items()}


def to_device(data, device):
    return dict(E=data['E'].to(device), X=data['X'].to(device), origins=data['origins'])


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    device = exp.device
    cell = CELLS[code]
    sel = load_split(exp, 'train', cell['history'], args.forecast_bs)
    test = load_split(exp, 'test', cell['history'], args.forecast_bs)
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    test_base = float(test['E'][test['origins'] - L].double().square().mean())
    if abs(test_base - official) > 1e-4:
        raise AssertionError(f'base cached={test_base} official={official}')
    del exp
    torch.cuda.empty_cache()
    cands = {(m, b, e, kk) for m in MODES for b in BETAS for e in ETAS for kk in KS}
    pools = args.pools.split(',')
    chosen, acc_v = {}, {}
    sel_d = to_device(sel, device)
    for pool in pools:
        ages = pool_ages(pool, cell['history'])
        acc_v[pool] = sums(sel_d, ages, device, BS[pool], cands, args.select_cases, seed)
        print(f'[select] {pool} n_demos {len(ages)}', flush=True)
    del sel_d, sel
    torch.cuda.empty_cache()
    for pool in pools:
        av = acc_v[pool]
        for m in MODES:
            for kk in KS:
                best = min(((b, e, sc) for b in BETAS for e in ETAS for sc in S_GRID),
                           key=lambda x: mse_at(av[m, x[0], x[1], kk], x[2]))
                chosen[f'{pool}:{m}:K{kk}'] = dict(pool=pool, mode=m, K=kk, beta=best[0], eta=best[1], s=best[2],
                                                   sel_mse=mse_at(av[m, best[0], best[1], kk], best[2]))
            bestk = min((x for x in chosen.values() if x['pool'] == pool and x['mode'] == m), key=lambda x: x['sel_mse'])
            chosen[f'{pool}:{m}:Kval'] = dict(bestk)
    test_d = to_device(test, device)
    del test
    result = {}
    base = None
    for pool in pools:
        ages = pool_ages(pool, cell['history'])
        mine = {f: x for f, x in chosen.items() if x['pool'] == pool}
        acc_t = sums(test_d, ages, device, BS[pool], {(x['mode'], x['beta'], x['eta'], x['K']) for x in mine.values()},
                     args.test_cases, seed + 1)
        a0 = next(iter(acc_t.values()))
        base = float(a0[0] / a0[3])
        for f, x in mine.items():
            m = mse_at(acc_t[x['mode'], x['beta'], x['eta'], x['K']], x['s'])
            result[f] = dict(x, n_demos=len(ages), test_mse=m, test_gain_pct=100 * (base - m) / base)
            print(f'[test] {f:18s} beta={x["beta"]} eta={x["eta"]} s={x["s"]} gain={result[f]["test_gain_pct"]:+.3f}%',
                  flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, history=cell['history'], tau_e=TAU_E, tau_y=TAU_Y, Ks=KS, etas=ETAS,
               betas=BETAS, select_split='train', select_cases=args.select_cases, checkpoint=str(checkpoint),
               test_cases=args.test_cases, test_base_full=test_base, test_base=base, families=result)
    output = Path(args.out_dir) / f'{code}_H{H}_L{L}_dense_multistep_s{seed}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T', choices=CELLS)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--pools', default='phase,daily,hourly')
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--select_cases', type=int, default=200000)
    p.add_argument('--test_cases', type=int, default=400000)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
