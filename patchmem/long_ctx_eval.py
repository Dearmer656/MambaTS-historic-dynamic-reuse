"""PAT-307 eval-only: lookback / horizon beyond the training range (L <= 1440, H <= 336). Trained checkpoints, test split, all windows.
Arms: A1 (R1), O (R1Obs), TF0 (R1Tefl hc=0), TF0r (R1Tefl hc=2, ETTh2 only).
  TF0 reads at most Tmax-1 = 29 lookback errors (its input size): for L > 1440 the errors of the LAST 29 slots are used, the backbone
      still reads the whole window (same RevIN statistics as every arm).
  O   'O'     retrieves over all T-1 slots (any L);
      'O29'   softmax restricted to the last 29 slots (what TF0 can see): O - O29 = the gain from retrieving beyond the training range.
      Also the attention mass on slots older than the last 29.
H > 336: O / A1 roll out natively, TF0 repeats its last correction block (extrap='last').
Sanity: L <= 1440 cells reproduce the training log's test MSE (checked by the reader), O29 == O when T-1 <= 29 (asserted).
usage (repo root, via sbatch): python patchmem/long_ctx_eval.py --data E|W --seed 3047 [--arms A1,O,TF0,TF0r] [--grid 1440:96,...]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.mech_eval import DATA
from patchmem.naive_fb_eval import load
from patchmem.obs_recent_eval import patch as patch_recent

DES = {'E': dict(A1='R1h0s{}', O='R1ObK0.5Q8h0s{}', TF0='R1TfnH0R64h0s{}', TF0r='R1TfnH2R64h0s{}', EP='R1ObK0.5Q8PVh0s{}'),
       'W': dict(A1='R1Wh0s{}', O='R1WObK0.5Q8h0s{}', TF0='R1WTfnH0R64h0s{}', EP='R1WObK0.5Q8PVh0s{}')}
ENV = dict(A1={}, O={'R1_OBS': '1', 'R1_MEM_QK_DIM': '8', 'R1_MEM_SLOT_KEEP': '0.5'},
           TF0={'R1_TEFL': 'native', 'R1_TEFL_HC': '0'}, TF0r={'R1_TEFL': 'native', 'R1_TEFL_HC': '2'},
           EP={'R1_OBS': '1', 'R1_OBS_VPATH': '1', 'R1_MEM_QK_DIM': '8', 'R1_MEM_SLOT_KEEP': '0.5'})
CLEAR = ('R1_TEFL', 'R1_TEFL_HC', 'R1_TEFL_RANK', 'R1_TEFL_FEED', 'R1_TEFL_EXTRAP')
NSLOT = 29                                                                              # training Tmax - 1


def truncate_tefl(m):
    """hc=0: errors of the last Tmax-1 slots only when the window is longer than Tmax patches (else the original error())."""
    f0 = m.error

    def err(x, xn, xo, yh, perm, means, stdev, lens=None):
        P, T = m.P, x.shape[1] // m.P
        if m.hc == 0 and T > m.tmax:
            assert lens is None
            k = m.tmax
            return f0(x[:, -k * P:], xn[:, -k * P:], xo[:, -k:], yh[:, -k:], perm, means, stdev, None)
        return f0(x, xn, xo, yh, perm, means, stdev, lens)
    m.error = err


@torch.no_grad()
def run_cell(e, arm, L, H, bs):
    m, dev, P = e.model, e.device, e.model.P
    ds = e._ds('test', L, H)
    ids = list(range(len(ds)))
    T = L // P
    variants = [(arm, None)] + ([('O29', NSLOT)] if arm == 'O' else [])
    se = {v: 0.0 for v, _ in variants}
    far, nfar, nel = 0.0, 0, 0
    for s0 in range(0, len(ids), bs):
        bx, by = e._batch(ds, ids[s0:s0 + bs])
        bx, y = bx.to(dev), by[:, -H:].to(dev)
        perm = m.scan_order(bx.shape[0], dev)
        for v, k in variants:
            if arm == 'O':
                m.obs_net.keep_k = k
                if k is None:
                    m.obs_net.mass_far = []
            se[v] += (m(bx, H // P, perm=perm)['pred'] - y).square().sum().item()
        if arm == 'O':
            m.obs_net.keep_k = None
        nel += y.numel()
    out = {v: se[v] / nel for v in se}
    if arm == 'O' and T - 1 <= NSLOT:
        assert abs(out['O29'] - out['O']) < 1e-6 * out['O'], out
    return out, len(ids)


def patch_far(net):
    """Record the attention mass on slots older than the last NSLOT (O, all slots)."""
    patch_recent(net)
    f1 = net.forward

    def fwd(h, keys, vals, mask=None):
        delta, a, g = f1(h, keys, vals, mask)
        if net.keep_k is None and a.shape[-1] > NSLOT:
            net.far_sum = getattr(net, 'far_sum', 0.0) + a[..., :-NSLOT].sum(-1).mean().item()
            net.far_n = getattr(net, 'far_n', 0) + 1
        return delta, a, g
    net.forward = fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=['E', 'W']); ap.add_argument('--seed', type=int, default=3047)
    ap.add_argument('--arms', default=None)
    ap.add_argument('--grid', default=','.join(f'{L}:{H}' for L in (1440, 2160, 2880) for H in (96, 336, 720)))
    ap.add_argument('--bs', type=int, default=32); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    arms = a.arms.split(',') if a.arms else [k for k in DES[a.data] if k != 'EP']
    base, ck, _, _ = DATA[a.data]
    out = a.out or f'patchmem/logs/long_ctx/{a.data}_s{a.seed}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = json.load(open(out)) if os.path.exists(out) else {}
    for arm in arms:
        for k in CLEAR:
            os.environ.pop(k, None)
        e = load(base, ck, DES[a.data][arm].format(a.seed), a.seed, ENV[arm])
        m = e.model
        if arm.startswith('TF0'):
            assert m.tmax == NSLOT + 1, m.tmax
            truncate_tefl(m)
        if arm == 'O':
            patch_far(m.obs_net)
        res.setdefault(arm, {})
        for g in a.grid.split(','):
            L, H = map(int, g.split(':'))
            if arm == 'O':
                m.obs_net.far_sum, m.obs_net.far_n = 0.0, 0
            mse, n = run_cell(e, arm, L, H, a.bs)
            rec = dict(mse=mse, windows=n)
            if arm == 'O':
                rec['far_mass'] = m.obs_net.far_sum / max(m.obs_net.far_n, 1)
            res[arm][f'{L}/{H}'] = rec
            print(f'[long_ctx {a.data} s{a.seed} {arm}] L={L} H={H} n={n} | ' + ' '.join(f'{v} {z:.4f}' for v, z in mse.items())
                  + (f' | attn mass beyond last {NSLOT} slots {rec["far_mass"]:.3f}' if arm == 'O' else ''), flush=True)
            json.dump(res, open(out, 'w'), indent=1)
        del e, m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
