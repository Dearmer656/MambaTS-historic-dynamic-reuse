"""PAT-307 eval-only: does O (R1Obs) select useful slots beyond the most recent ones? Trained O checkpoint, every (L, H) cell, all windows.
Variants (compare with patchmem/obs_recent_eval.py recentK at the same K):
  normal          all T-1 slots (the trained model)
  topK            per head, softmax restricted to the K slots with the largest logits (K = 1, 2, 4), renormalised
Top-1 lag histogram of the normal model: lag = (T-1) - 1 - argmax slot index (lag 0 = the last lookback patch's error), binned
0 | 1 | 2-3 | 4-7 | 8+, fraction over heads, rollout steps and windows; lag_full: the top-1 fraction and the mean attention per
single lag (index = lag in patches), to look for periodic peaks.
Sanity: the re-implemented forward equals ObsMem.forward (first batch of every cell); topK with K = T-1 equals normal.
usage (repo root, via sbatch): python patchmem/obs_topk_eval.py --data E|W --seed 3047 [--grid 384:96,...]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.mech_eval import DATA
from patchmem.naive_fb_eval import load
from patchmem.obs_recent_eval import ENV_O, DES_O

KS = (1, 2, 4)
BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 10 ** 9))


def patch(net):
    """Replace ObsMem.forward (eval, no mask, learned attention only): net.topk (None = all slots) keeps, per head, the K largest
    logits; net.lag accumulates the top-1 lag histogram when topk is None. Same arithmetic as ObsMem.forward otherwise."""
    assert not net.uniform

    def fwd(h, keys, vals, mask=None):
        assert mask is None
        b, M = keys.shape[:2]
        hq = h.detach().flatten(1)
        q = net.Wq(hq).reshape(b, net.nh, net.dh)
        logit = torch.einsum('bhd,bmhd->bhm', q, keys) / (net.dh ** 0.5 * net.log_tau.exp())[None, :, None]
        if net.topk is not None and net.topk < M:
            thr = logit.topk(net.topk, -1).values[..., -1:]
            logit = logit.masked_fill(logit < thr, float('-inf'))
        a = torch.softmax(logit, -1)
        if net.topk is None:
            lag = (M - 1) - logit.argmax(-1)
            for i, (lo, hi) in enumerate(BINS):
                net.lag[i] += ((lag >= lo) & (lag <= hi)).sum().item()
            net.lag[-1] += lag.numel()
            net.lag_full[:M] += torch.bincount(lag.flatten(), minlength=M).double().cpu()        # top-1 count per lag
            net.mass_full[:M] += a.double().sum((0, 1)).flip(0).cpu()                         # attention mass per lag
            net.n_full[:M] += a.shape[0] * a.shape[1]                                          # (b, nh) rows that have this lag
        m = torch.einsum('bhm,bmcp->bhcp', a, vals)
        g = torch.sigmoid(net.Wg(hq)).unsqueeze(-1)
        return g * (net.W[None] * m).sum(1), a, g
    net._orig, net.forward, net.topk = net.forward, fwd, None


@torch.no_grad()
def run_cell(e, flag, L, H, bs=64):
    m, dev, P = e.model, e.device, e.model.P
    net = m.obs_net
    ds = e._ds(flag, L, H)
    ids = list(range(len(ds)))
    T = L // P
    variants = [('normal', None)] + [(f'top{k}', k) for k in KS] + [('allK', T - 1)]
    se = {v: 0.0 for v, _ in variants}
    nel = 0
    net.lag = [0] * (len(BINS) + 1)
    net.lag_full, net.mass_full, net.n_full = (torch.zeros(T, dtype=torch.float64) for _ in range(3))
    for s0 in range(0, len(ids), bs):
        bx, by = e._batch(ds, ids[s0:s0 + bs])
        bx, y = bx.to(dev), by[:, -H:].to(dev)
        perm = m.scan_order(bx.shape[0], dev)
        if s0 == 0:                                                                     # patched forward == ObsMem.forward
            f1, net.forward = net.forward, net._orig
            ref = m(bx, H // P, perm=perm)['pred']
            net.forward = f1
            net.topk = None
            d = (m(bx, H // P, perm=perm)['pred'] - ref).abs().max().item()
            assert d < 1e-4 * ref.abs().max().item(), d
        for v, k in variants:
            net.topk = k
            se[v] += (m(bx, H // P, perm=perm)['pred'] - y).square().sum().item()
        net.topk = None
        nel += y.numel()
    mse = {v: se[v] / nel for v in se}
    assert abs(mse['allK'] - mse['normal']) < 1e-6 * max(mse['normal'], 1e-6), (mse['allK'], mse['normal'])
    lag = {f'{lo}-{hi}' if hi < 10 ** 9 else f'{lo}+': net.lag[i] / net.lag[-1] for i, (lo, hi) in enumerate(BINS)}
    full = dict(top1=(net.lag_full / net.lag_full.sum()).tolist(), mass=(net.mass_full / net.n_full.clamp_min(1)).tolist())
    return mse, lag, full, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=['E', 'W']); ap.add_argument('--seed', type=int, default=3047)
    ap.add_argument('--tag', default='', help='output file suffix (keeps earlier full-grid results)')
    ap.add_argument('--p24', action='store_true', help='ETTh2 patch_len 24 checkpoints (patchmem/slurm/p24_job.sbatch)')
    ap.add_argument('--grid', default=','.join(f'{L}:{H}' for L in (384, 720, 1440) for H in (96, 192, 336)))
    a = ap.parse_args()
    base, ck, _, _ = DATA[a.data]
    if a.p24:
        assert a.data == 'E'
        base = base + ['--patch_len', '24', '--stride', '24']
        e = load(base, ck, f'R1ObK0.5Q8P24h0s{a.seed}', a.seed, dict(ENV_O, R1_HMIN='4', R1_HMAX='14'))
        assert e.model.P == 24
    else:
        e = load(base, ck, DES_O[a.data].format(a.seed), a.seed, ENV_O)
    patch(e.model.obs_net)
    out = f'patchmem/logs/obs_topk/{a.data}{"_P24" if a.p24 else ""}_s{a.seed}{a.tag}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = {}
    for g in a.grid.split(','):
        L, H = map(int, g.split(':'))
        res[f'{L}/{H}'] = {}
        for flag in ('val', 'test'):
            mse, lag, full, n = run_cell(e, flag, L, H)
            res[f'{L}/{H}'][flag] = dict(mse=mse, top1_lag=lag, lag_full=full, windows=n)
            print(f'[obs_topk {a.data} s{a.seed} {flag}] L={L} H={H} n={n} | ' + ' '.join(f'{v} {z:.4f}' for v, z in mse.items() if v != 'allK')
                  + ' | top1 lag ' + ' '.join(f'{k}:{z:.3f}' for k, z in lag.items()), flush=True)
            json.dump(dict(data=a.data, seed=a.seed, grid=res), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
