"""PAT-307 offline probe: VOTING over in-window ICL experts of different lags (no model training; A1 seed-3047 checkpoint, frozen).
Experts = patchmem/icl_hedge.py experts(): per lead k, single g (G = {g}) and pair g (G = {k, g}) in-window linear ridge (lam = 1),
recomputed inside every window. Lag set: 'single' or 'all' (single + pair). Aggregations (no weights learnt anywhere):
  vmean    uniform average of the lag experts' corrections
  vmedian  elementwise median
  vagree   elementwise sign agreement a = |2 f - 1| (f = fraction of lag experts with delta > 0); where a >= thr use the vmedian
           value, elsewhere fall back to recent8 (mean of the 8 most recent in-window errors); thr in THRS
Family / thr chosen on val per lead (agg is reported separately for each). Shuffle control (sh_): the same with every expert's demo
INPUTS permuted (icl_lag v3 null), NSH draws averaged; recent8 / mean are input-free, unchanged under the null.
Reported: absolute test MSEred of the real and the shuffle run (not only the difference), and recent8 / mean baselines.
Pre-registered (ETTh2, A1, L = 1440, test, mean over leads 4-7), for the best of the three aggregations chosen on val:
  vote - recent8 >= 0.02   AND   vote - sh_vote >= 0.02.   Weather and L = 720 secondary.
usage: python patchmem/icl_vote.py --data E|W [--L 1440,720] [--stride 1]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from patchmem.icl_hedge import expert_preds, P, NP, SEED, dev

THRS = (0.5, 0.7, 0.9)
AGGS = ('vmean', 'vmedian', 'vagree')


def votes(names, pr):
    """pr (N, K, C, P) -> {(agg, cfg): delta (N, C, P)}."""
    rec = pr[:, names.index('recent8')]
    out = {}
    for fam in ('single', 'all'):
        idx = [i for i, n in enumerate(names) if n.startswith('single') or (fam == 'all' and n.startswith('pair'))]
        E = pr[:, idx]
        med = E.median(1).values
        out[('vmean', fam)] = E.mean(1)
        out[('vmedian', fam)] = med
        a = ((E > 0).float().mean(1) * 2 - 1).abs()
        for thr in THRS:
            out[('vagree', f'{fam}_t{thr}')] = torch.where(a >= thr, med, rec)
    return out


def main():
    from patchmem.diag_rollout import official_args
    from patchmem.resid_cov import CFG, H, split_x, a1_resid
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W'])
    ap.add_argument('--L', default='1440,720'); ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--chunk', type=int, default=128); ap.add_argument('--nsh', type=int, default=3)
    a_ = ap.parse_args()
    cf = CFG[a_.data]; res = {}
    os.environ.pop('R1_OBS', None)
    from patchmem.exp_r1 import Exp_R1
    ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {cf["r1des"]} --seed {SEED}').split())
    ex = Exp_R1(ar)
    ex.model.load_state_dict(torch.load(f'checkpoints/{cf["r1ck"]}/checkpoint.pth', map_location=dev))
    gen = torch.Generator(device=dev).manual_seed(0)
    for L in [int(v) for v in a_.L.split(',')]:
        J = L // P - 1
        R = {}
        for sp in ('val', 'test'):
            X = split_x(ar, L, sp)
            origins = list(range(L + J * P, len(X) - H + 1))[::a_.stride]
            R[sp] = a1_resid(ex, X, L, origins, dev, mmax=J)
        print(f'===== {a_.data} A1 L={L} (J={J}) test {len(R["test"][0])} windows | test MSEred real / shuffle (config chosen on val per lead)', flush=True)
        tab = {nm: np.full((3, NP), np.nan) for nm in AGGS + ('best',)}                # rows: real, sh, val
        base = {'recent8': np.zeros(NP), 'mean': np.zeros(NP)}
        cfgs = {nm: [] for nm in AGGS + ('best',)}
        for k in range(1, NP + 1):
            sc = {}
            for sp in ('val', 'test'):
                names, r, pr = expert_preds(*R[sp], k, gen, a_.chunk, a_.nsh)
                ss = (r ** 2).sum().item()
                m = lambda d: 1 - ((r - d) ** 2).sum().item() / ss
                real = {key: m(d) for key, d in votes(names, pr[0]).items()}
                sh = {}
                for p in pr[1:]:
                    for key, d in votes(names, p).items():
                        sh[key] = sh.get(key, 0) + m(d) / (len(pr) - 1)
                sc[sp] = (real, sh)
                if sp == 'test':
                    base['recent8'][k - 1] = m(pr[0][:, names.index('recent8')]); base['mean'][k - 1] = m(pr[0][:, names.index('mean')])
            for nm in AGGS:
                keys = [key for key in sc['val'][0] if key[0] == nm]
                b = max(keys, key=lambda key: sc['val'][0][key])
                tab[nm][:, k - 1] = (sc['test'][0][b], sc['test'][1][b], sc['val'][0][b]); cfgs[nm].append(b[1])
            b = max(sc['val'][0], key=lambda key: sc['val'][0][key])
            tab['best'][:, k - 1] = (sc['test'][0][b], sc['test'][1][b], sc['val'][0][b]); cfgs['best'].append(f'{b[0]}:{b[1]}')
        f = lambda v: ' '.join(f'{x:+.3f}' for x in v)
        for nm in AGGS + ('best',):
            t = tab[nm]
            print(f'  {nm:8s} real {f(t[0])} | sh {f(t[1])} | L4-7 real {t[0, 3:].mean():+.3f} sh {t[1, 3:].mean():+.3f} | cfg {cfgs[nm]}', flush=True)
        for nm, v in base.items():
            print(f'  {nm:8s} real {f(v)} | L4-7 {v[3:].mean():+.3f}', flush=True)
        t = tab['best']
        print(f'  criterion (best on val): vote - recent8 {t[0, 3:].mean() - base["recent8"][3:].mean():+.3f} | vote - sh {t[0, 3:].mean() - t[1, 3:].mean():+.3f}', flush=True)
        res[f'A1_L{L}'] = dict(tab={nm: v.tolist() for nm, v in tab.items()}, cfgs=cfgs, base={nm: v.tolist() for nm, v in base.items()})
        json.dump(res, open(f'patchmem/logs/server/icl_vote_{a_.data}_s{SEED}.json', 'w'))
    print('DONE')


if __name__ == '__main__':
    main()
