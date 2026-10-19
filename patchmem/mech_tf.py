"""PAT-304: is the decay of the memory / error-feedback gain along the rollout caused by the information (lookback errors lose relevance with
the distance) or by the rollout itself (the model is fed its own forecasts)? Eval-only on trained A1 / Q8+drop50 checkpoints.
Per (L, H), per future patch k, test MSE of A1, Q8 and the non-learned error feedback on A1 (A1 forecast + beta * b, beta one scalar fitted on
val per condition, b = level K=1 / profile K=3 / profile K=0 of the lookback one-step errors, see patchmem/mech_eval.py), under two conditions:
  rollout  the model is fed its own previous forecast (the normal test)
  tf       teacher forcing: the model is fed the TRUE previous future patch (patchmem/diag_rollout.run mode 'tf'), so every future patch is a
           one-step prediction from the true past; the memory still reads only lookback slots. Patch k = 0 is identical in both conditions.
usage (repo root): python patchmem/mech_tf.py --data E|W|T --seed 3047 [--grid 1440:336,384:336] [--stride 1]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import run
from patchmem.mech_eval import DATA, Q8ENV, load, hist_err, fb

SPECS = [('level', 1), ('profile', 3), ('profile', 0)]


@torch.no_grad()
def cell(eA, eQ, flag, L, H, stride, fit=None):
    mA, mQ, P, dev = eA.model, eQ.model, eA.model.P, eA.device
    ds = eA._ds(flag, L, H)
    ids = list(range(0, len(ds), stride))
    n_out = H // P
    SK, bsum, nel = {}, {}, 0
    acc = lambda k, e: SK.__setitem__(k, SK.get(k, 0.0) + e.reshape(e.shape[0], n_out, -1).square().sum((0, 2)).double().cpu())
    for s0 in range(0, len(ids), 64):
        bx, by = eA._batch(ds, ids[s0:s0 + 64])
        bx, y = bx.to(dev), by[:, -H:].to(dev)
        eh = hist_err(bx, mA(bx, n_out)['hist'], P)
        nel += y.numel() / n_out
        for cond in ('rollout', 'tf'):
            md = 'normal' if cond == 'rollout' else 'tf'
            pA, pQ = run(mA, bx, y, n_out, md), run(mQ, bx, y, n_out, md)
            acc(f'{cond}/A1', pA - y); acc(f'{cond}/Q8', pQ - y)
            r = (y - pA).reshape(bx.shape[0], n_out, P, -1)
            for kind, K in SPECS:
                bb = fb(eh, K, kind)[:, None].expand_as(r)
                key = f'{cond}/fb_{kind}_K{K}'
                t = bsum.setdefault(key, [0.0, 0.0])
                t[0] += (bb * r).sum().item(); t[1] += bb.square().sum().item()
                if fit is not None:
                    acc(key, r - fit[key] * bb)
    beta = {k: v[0] / max(v[1], 1e-12) for k, v in bsum.items()}
    return {k: (v / nel).tolist() for k, v in SK.items()}, beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=list(DATA)); ap.add_argument('--seed', type=int, default=3047)
    ap.add_argument('--grid', default='1440:336,384:336'); ap.add_argument('--stride', type=int, default=1)
    a = ap.parse_args()
    base, ck, dA, dQ = DATA[a.data]
    eA = load(base, ck, dA.format(a.seed), a.seed, {})
    eQ = load(base, ck, dQ.format(a.seed), a.seed, Q8ENV)
    res = {}
    for g in a.grid.split(','):
        L, H = map(int, g.split(':'))
        _, beta = cell(eA, eQ, 'val', L, H, a.stride)
        pm, _ = cell(eA, eQ, 'test', L, H, a.stride, fit=beta)
        res[f'{L}/{H}'] = dict(patch_mse=pm, beta_val=beta)
        print(f'[tf {a.data} s{a.seed}] L={L} H={H}', flush=True)
        for cond in ('rollout', 'tf'):
            A = np.array(pm[f'{cond}/A1'])
            print(f'  {cond}: A1 mse        ' + ' '.join(f'{v:.4f}' for v in A), flush=True)
            for k in ('Q8', 'fb_level_K1', 'fb_profile_K3', 'fb_profile_K0'):
                v = np.array(pm[f'{cond}/{k}'])
                print(f'  {cond}: {k:13s} gain% ' + ' '.join(f'{100 * (A[i] - v[i]) / A[i]:+6.1f}' for i in range(len(A))), flush=True)
        os.makedirs('patchmem/logs/mech', exist_ok=True)
        json.dump(dict(data=a.data, seed=a.seed, stride=a.stride, grid=res), open(f'patchmem/logs/mech/tf_{a.data}_s{a.seed}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
