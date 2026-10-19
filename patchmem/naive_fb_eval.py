"""PAT-304: A1 + naive observation-error feedback, evaluated next to the trained arms on the SAME windows (eval only, no training).
For every (L, H) cell: A1 (the R1 checkpoint), A1 + b with b built from A1's own next-patch errors inside the lookback
(e_t = x_{t+1} - hist_t, last K errors, K = 0 means all; 'level' = mean over patch and errors, 'profile' = per position in the patch),
forecast = pred(A1) + beta * b repeated over the horizon, ONE scalar beta per (kind, K) least-squares fitted on VAL and applied to TEST
(no test-set choice of beta; the reported K is NOT selected here, all K are listed). Optional trained arms on the same windows:
--arms Q8 O U (checkpoints R1MK0.5Q8c1h0s*, R1ObK0.5Q8h0s*, R1MUK0.5c1h0s*). Check: beta = 0 must reproduce the A1 test MSE of the
training script (same windows, stride 1).
usage (repo root, via sbatch): python patchmem/naive_fb_eval.py --data E|W --seed 3047 [--grid 384:96,...] [--arms Q8,O]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.mech_eval import DATA, hist_err, fb
from patchmem.diag_rollout import official_args

ARMS = {  # des template per data, env
    'Q8': ({'E': 'R1MK0.5Q8c1h0s{}', 'W': 'R1WMK0.5Q8c1h0s{}'}, {'R1_MEM': '1', 'R1_MEM_QK_DIM': '8', 'R1_MEM_SLOT_KEEP': '0.5'}),
    'U': ({'E': 'R1MUK0.5c1h0s{}', 'W': 'R1WMUK0.5c1h0s{}'}, {'R1_MEM': '1', 'R1_MEM_UNIFORM': '1', 'R1_MEM_SLOT_KEEP': '0.5'}),
    'O': ({'E': 'R1ObK0.5Q8h0s{}', 'W': 'R1WObK0.5Q8h0s{}'}, {'R1_OBS': '1', 'R1_MEM_QK_DIM': '8', 'R1_MEM_SLOT_KEEP': '0.5'}),
}
ENVKEYS = ('R1_MEM', 'R1_MEM_QK_DIM', 'R1_MEM_SLOT_KEEP', 'R1_MEM_UNIFORM', 'R1_OBS')


def load(base, ck, des, seed, env):
    for k in ENVKEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    from patchmem.exp_r1 import Exp_R1
    args = official_args(base + ['--seed', str(seed), '--des', des])
    exp = Exp_R1(args)
    m = exp.model
    path = os.path.join(args.checkpoints, ck.format(des), 'checkpoint.pth')
    m.load_state_dict(torch.load(path, map_location=exp.device))     # strict: a wrong arm/config fails here instead of evaluating junk
    m.eval(); m.reset_ids_shuffle(); exp._fix_eval_order()
    return exp


@torch.no_grad()
def run_cell(eA, others, flag, L, H, Ks, fit=None, bs=64):
    mA, P, dev = eA.model, eA.model.P, eA.device
    ds = eA._ds(flag, L, H)
    ids = list(range(len(ds)))
    n_out = H // P
    S, ls = {}, {}
    add = lambda d, k, v: d.__setitem__(k, d.get(k, 0.0) + float(v))
    nel = 0
    for s0 in range(0, len(ids), bs):
        bx, by = eA._batch(ds, ids[s0:s0 + bs])
        bx, y = bx.to(dev), by[:, -H:].to(dev)
        o = mA(bx, n_out)
        pA = o['pred']; b, c = pA.shape[0], pA.shape[-1]
        nel += pA.numel()
        add(S, 'A1', (pA - y).square().sum())
        r = (y - pA).reshape(b, n_out, P, c)
        eh = hist_err(bx, o['hist'], P)
        for K in Ks:
            for kind in ('level', 'profile'):
                bb = fb(eh, K, kind)[:, None].expand(b, n_out, P, c)
                key = f'A1+{kind}_K{K}'
                ls.setdefault(key, [0.0, 0.0])
                ls[key][0] += (bb * r).sum().item(); ls[key][1] += bb.square().sum().item()
                if fit is not None:
                    add(S, key, (r - fit[key] * bb).square().sum())
        for name, eo in others.items():
            bo, yo = eo._batch(eo._ds(flag, L, H), ids[s0:s0 + bs]) if eo is not eA else (bx, by)
            po = eo.model(bo.to(dev), n_out)['pred']
            add(S, name, (po - yo[:, -H:].to(dev)).square().sum())
    return {k: v / nel for k, v in S.items()}, {k: v[0] / max(v[1], 1e-12) for k, v in ls.items()}, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=['E', 'W']); ap.add_argument('--seed', type=int, default=3047)
    ap.add_argument('--grid', default=','.join(f'{L}:{H}' for L in (384, 720, 1440) for H in (96, 192, 336)))
    ap.add_argument('--K', default='1,3,0'); ap.add_argument('--arms', default='Q8,O'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    base, ck, dA, _ = DATA[a.data]
    dA = {'E': 'R1h0s{}', 'W': 'R1Wh0s{}'}[a.data]                      # full-budget A1 (mech_eval's W entry is the 1000-iter local run)
    eA = load(base, ck, dA.format(a.seed), a.seed, {})
    others = {}
    for n in [x for x in a.arms.split(',') if x]:
        des, env = ARMS[n]
        others[n] = load(base, ck, des[a.data].format(a.seed), a.seed, env)
    Ks = [int(k) for k in a.K.split(',')]
    out = a.out or f'patchmem/logs/naive/{a.data}_s{a.seed}'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = {}
    for g in a.grid.split(','):
        L, H = map(int, g.split(':'))
        _, beta, nv = run_cell(eA, {}, 'val', L, H, Ks)
        r, _, nt = run_cell(eA, others, 'test', L, H, Ks, fit=beta)
        r['beta_val'] = beta; r['windows'] = nt
        res[f'{L}/{H}'] = r
        A = r['A1']
        print(f'[naive {a.data} s{a.seed}] L={L} H={H} windows {nt} | A1 {A:.4f} | ' + ' '.join(f'{n} {r[n]:.4f}' for n in others) + ' | ' +
              ' '.join(f'{k[3:]} {r[k]:.4f}' for k in sorted(r) if k.startswith('A1+')), flush=True)
        print('   beta(val): ' + ' '.join(f'{k[3:]} {v:.2f}' for k, v in sorted(beta.items())), flush=True)
        json.dump(dict(data=a.data, seed=a.seed, grid=res), open(out + '.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
