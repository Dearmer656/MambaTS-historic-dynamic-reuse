"""PAT-307 eval-only: how much does O (R1Obs) use the distant lookback errors? Trained O checkpoint, every (L, H) cell, all windows.
Variants: normal (all T-1 slots) | recentK (softmax restricted to the last K slots, K = 1, 2, 4; K = 1 -> no selection, a = 1).
Also the attention mass the normal model puts on the last 1 / 2 / 4 slots (mean over heads, rollout steps, windows).
Sanity: restricting to K = T-1 slots must equal normal exactly, and normal matches the training log's test MSE.
usage (repo root, via sbatch): python patchmem/obs_recent_eval.py --data E|W --seed 3047 [--grid 384:96,...]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.mech_eval import DATA
from patchmem.naive_fb_eval import load

KS = (1, 2, 4)
ENV_O = {'R1_OBS': '1', 'R1_MEM_QK_DIM': '8', 'R1_MEM_SLOT_KEEP': '0.5'}
DES_O = {'E': 'R1ObK0.5Q8h0s{}', 'W': 'R1WObK0.5Q8h0s{}'}


def patch(net):
    """Wrap ObsMem.forward: net.keep_k (None = all slots) restricts the softmax to the last K slots; net.mass accumulates the
    attention on the last 1/2/4 slots."""
    f0 = net.forward

    def fwd(h, keys, vals, mask=None):
        b, M = keys.shape[:2]
        if net.keep_k is not None:
            keep = (torch.arange(M, device=keys.device) >= M - net.keep_k)[None, :].expand(b, M)
            mask = keep if mask is None else mask & keep
        delta, a, g = f0(h, keys, vals, mask)
        if net.keep_k is None:
            for k in KS:
                net.mass[k] = net.mass.get(k, 0.0) + a[..., -k:].sum(-1).mean().item()
            net.mass['n'] = net.mass.get('n', 0) + 1
        return delta, a, g
    net.forward, net.keep_k, net.mass = fwd, None, {}


@torch.no_grad()
def run_cell(e, flag, L, H, bs=64):
    m, dev, P = e.model, e.device, e.model.P
    net = m.obs_net
    ds = e._ds(flag, L, H)
    ids = list(range(len(ds)))
    T = L // P
    variants = [('normal', None)] + [(f'recent{k}', k) for k in KS] + [('allK', T - 1)]
    se = {v: 0.0 for v, _ in variants}
    nel = 0
    net.mass = {}
    for s0 in range(0, len(ids), bs):
        bx, by = e._batch(ds, ids[s0:s0 + bs])
        bx, y = bx.to(dev), by[:, -H:].to(dev)
        perm = m.scan_order(bx.shape[0], dev)
        for v, k in variants:
            net.keep_k = k
            se[v] += (m(bx, H // P, perm=perm)['pred'] - y).square().sum().item()
        net.keep_k = None
        nel += y.numel()
    mse = {v: se[v] / nel for v in se}
    assert abs(mse['allK'] - mse['normal']) < 1e-6 * max(mse['normal'], 1e-6), (mse['allK'], mse['normal'])
    mass = {f'last{k}': net.mass[k] / net.mass['n'] for k in KS}
    return mse, mass, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=['E', 'W']); ap.add_argument('--seed', type=int, default=3047)
    ap.add_argument('--grid', default=','.join(f'{L}:{H}' for L in (384, 720, 1440) for H in (96, 192, 336)))
    a = ap.parse_args()
    base, ck, _, _ = DATA[a.data]
    e = load(base, ck, DES_O[a.data].format(a.seed), a.seed, ENV_O)
    patch(e.model.obs_net)
    out = f'patchmem/logs/obs_recent/{a.data}_s{a.seed}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = {}
    for g in a.grid.split(','):
        L, H = map(int, g.split(':'))
        res[f'{L}/{H}'] = {}
        for flag in ('val', 'test'):
            mse, mass, n = run_cell(e, flag, L, H)
            res[f'{L}/{H}'][flag] = dict(mse=mse, mass=mass, windows=n)
            print(f'[obs_recent {a.data} s{a.seed} {flag}] L={L} H={H} n={n} | ' + ' '.join(f'{v} {z:.4f}' for v, z in mse.items() if v != 'allK')
                  + ' | attn mass ' + ' '.join(f'{k} {z:.3f}' for k, z in mass.items()), flush=True)
            json.dump(dict(data=a.data, seed=a.seed, grid=res), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
