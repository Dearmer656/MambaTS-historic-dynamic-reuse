"""PAT-315 training-free: how much observed y should the key carry next to the error?

Same cases/demos/protocol as eval_raw_sim (Traffic H192, 14 same-phase demos, cos similarities,
tau and s selected on a fixed 1M-case train subset, test once).  Components (query at t, demo at s_j):
  e      cos(E_{t-H}, E_{s_j-H})                          error over [t-H, t)
  y24    cos(y[t-24:t], y[s_j-24:s_j])                    past day
  y96    cos(y[t-96:t], ...)                              backbone's current input (L = 96)
  y192   cos(y[t-H:t], ...)                               same window as the error
  y96c / y192c  as above after removing each window's own mean (shape only)
  cat3   one cos on the concatenation x_o = [y[o-H-L:o-H], y[o-H:o], E_{o-H}] (old forecast's input, the window
         it covered, its error): one-layer W1 one-step-GD ICL in the exp-kernel = softmax attention, value E_{s_j}
  cat3n  same concatenation after l2-normalising each of the three segments
e_hat = s * sum_j softmax_j(sum_c tau_c sim_c) e_j.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_error_path import CELLS, H, L, all_cases, demo_ages, load_split
from patchmem.eval_raw_sim import S_GRID, mse_at

TAUS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
TAUS3 = (0.5, 1.0, 2.0, 4.0, 8.0)
FAMILIES = {'mean': (), 'e+y24': ('e', 'y24'), 'e+y96': ('e', 'y96'), 'e+y192': ('e', 'y192'),
            'e+y96c': ('e', 'y96c'), 'e+y192c': ('e', 'y192c'), 'e+y24+y192': ('e', 'y24', 'y192'),
            'y192': ('y192',), 'y96': ('y96',), 'cat3': ('cat3',), 'cat3n': ('cat3n',)}
YLEN = {'y24': 24, 'y96': 96, 'y192': H, 'y96c': 96, 'y192c': H}


def grid(comps):
    if not comps:
        return [()]
    taus = TAUS3 if len(comps) == 3 else TAUS
    return list(itertools.product(taus, repeat=len(comps)))


def combos():
    return sorted({(f, tau) for f, comps in FAMILIES.items() for tau in grid(comps)})


def cos(a, b):
    return F.cosine_similarity(a[:, None], b, dim=-1, eps=1e-8).double()


@torch.no_grad()
def sums(data, ages, device, bs, cands, max_cases=None, seed=0):
    E, X = data['E'], data['X']
    ag = torch.as_tensor(ages)
    kmax = H + L                                                      # y back to o-H-L
    k = torch.arange(kmax)
    t_all, c_all = all_cases(data, max_cases, seed)
    acc = {c: torch.zeros(3, dtype=torch.float64, device=device) for c in cands}
    n = 0
    for i in range(0, len(t_all), bs):
        t, c = t_all[i:i + bs], c_all[i:i + bs]
        s = t[:, None] - ag[None]
        cc = c[:, None]
        q = E[t - H - L, :, c].to(device)
        target = E[t - L, :, c].to(device).double()
        K = E[s - H - L, :, cc].to(device)
        V = E[s - L, :, cc].to(device).double()
        yq = X[(t - 1)[:, None] - k, c[:, None]].to(device)          # B kmax, index 0 = y_{t-1}
        ys = X[(s - 1)[..., None] - k, cc[..., None]].to(device)     # B n kmax
        S = {'e': cos(q, K)}
        y1q, y2q = yq[:, H:H + L], yq[:, :H]                            # y[t-H-L:t-H], y[t-H:t] (reversed time order)
        y1s, y2s = ys[..., H:H + L], ys[..., :H]
        S['cat3'] = cos(torch.cat([y1q, y2q, q], -1), torch.cat([y1s, y2s, K], -1))
        nz = lambda a: F.normalize(a, dim=-1)
        S['cat3n'] = cos(torch.cat([nz(y1q), nz(y2q), nz(q)], -1), torch.cat([nz(y1s), nz(y2s), nz(K)], -1))
        for name, ln in YLEN.items():
            a, b = yq[:, :ln], ys[..., :ln]
            if name.endswith('c'):
                a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
            S[name] = cos(a, b)
        tt = target.square().sum()
        for (f, tau) in cands:
            logit = sum((tv * S[cn] for tv, cn in zip(tau, FAMILIES[f])), torch.zeros_like(S['e']))
            eh = (torch.softmax(logit, -1)[..., None] * V).sum(1)
            acc[f, tau] += torch.stack([tt, (target * eh).sum(), eh.square().sum()])
        n += target.numel()
    return {c: torch.cat([a.cpu(), torch.tensor([float(n)], dtype=torch.float64)]) for c, a in acc.items()}


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    device = exp.device
    cell = CELLS[code]
    ages = demo_ages('phase', cell['history'], cell['period'], cell['stride'])
    sel = load_split(exp, 'train', cell['history'], args.forecast_bs)
    test = load_split(exp, 'test', cell['history'], args.forecast_bs)
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    test_base = float(test['E'][test['origins'] - L].double().square().mean())
    if abs(test_base - official) > 1e-4:
        raise AssertionError(f'base cached={test_base} official={official}')
    del exp
    torch.cuda.empty_cache()
    cands = combos()
    acc_v = sums(sel, ages, device, args.bs, cands, args.select_cases, seed)
    del sel
    chosen = {}
    for f in FAMILIES:
        best = min(((tau, s) for (ff, tau) in cands if ff == f for s in S_GRID),
                   key=lambda x: mse_at(acc_v[f, x[0]], x[1]))
        chosen[f] = dict(tau=best[0], s=best[1], sel_mse=mse_at(acc_v[f, best[0]], best[1]))
    acc_t = sums(test, ages, device, args.bs, [(f, x['tau']) for f, x in chosen.items()])
    a0 = acc_t['mean', ()]
    base = float(a0[0] / a0[3])
    assert abs(base - test_base) < 1e-6, (base, test_base)
    result = {}
    for f, x in chosen.items():
        m = mse_at(acc_t[f, x['tau']], x['s'])
        result[f] = dict(x, comps=FAMILIES[f], test_mse=m, test_gain_pct=100 * (base - m) / base)
        print(f'[test] {f:12s} tau={x["tau"]} s={x["s"]} gain={result[f]["test_gain_pct"]:+.3f}%', flush=True)
    out = dict(dataset=code, seed=seed, H=H, ages=ages, ylen=YLEN, taus=TAUS, taus3=TAUS3, select_split='train',
               select_cases=args.select_cases, checkpoint=str(checkpoint), test_base=base, families=result)
    output = Path(args.out_dir) / (f'{code}_H{H}_phase_rawylen2_s{seed}.json' if L == 96 else f'{code}_H{H}_L{L}_phase_rawylen2_s{seed}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T', choices=CELLS)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--bs', type=int, default=2048)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--select_cases', type=int, default=1000000)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
