"""PAT-315 training-free control: softmax weighting of the same-phase demo errors with
similarities computed directly on raw error / raw y (no learned map).

Same cases, demos and time axis as eval_error_path.py (Traffic H192, 14 same-phase demos):
  sim_e(j) = sim(q, k_j),            q = E_{t-H}, k_j = E_{s_j-H}               (H-dim errors)
  sim_y(j) = sim(y_{t-W..t-1}, y_{s_j-W..s_j-1}), W = 24                       (raw scaled y)
  sim in {cos, l2}: cos(a, b) or -|a - b|^2 / dim
  e_hat = s * sum_j softmax_j(tau_e sim_e + tau_y sim_y) e_j
Families (per sim type): mean (tau = 0), e (tau_y = 0), y (tau_e = 0), e+y (both > 0).
tau_e, tau_y and s are chosen per family on the full official val split; test is scored once.
MSE(s) is exact from three sums per tau pair (sum t^2, sum t.e_hat, sum e_hat^2).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_error_path import CELLS, H, L, all_cases, demo_ages, load_split

W = 24
TAUS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
S_GRID = tuple(round(0.05 * k, 2) for k in range(1, 41))      # 0.05 .. 2.0
SIMS = ('cos', 'l2')


def sim(a, b, kind):
    """a: B d, b: B n d -> B n."""
    if kind == 'cos':
        return F.cosine_similarity(a[:, None], b, dim=-1, eps=1e-8)
    return -(a[:, None] - b).square().mean(-1)


@torch.no_grad()
def sums(data, ages, device, bs, combos):
    """For every (kind, tau_e, tau_y): [sum t^2, sum t.e_hat, sum e_hat^2, count]."""
    E, X = data['E'], data['X']
    ag = torch.as_tensor(ages)
    k = torch.arange(W)
    t_all, c_all = all_cases(data)
    acc = {c: torch.zeros(4, dtype=torch.float64, device=device) for c in combos}
    for i in range(0, len(t_all), bs):
        t, c = t_all[i:i + bs], c_all[i:i + bs]
        s = t[:, None] - ag[None]
        cc = c[:, None]
        q = E[t - H - L, :, c].to(device)
        target = E[t - L, :, c].to(device).double()
        K = E[s - H - L, :, cc].to(device)
        V = E[s - L, :, cc].to(device).double()
        yq = X[(t - 1)[:, None] - k, c[:, None]].to(device)          # B W   (ends at t-1)
        ys = X[(s - 1)[..., None] - k, cc[..., None]].to(device)     # B n W (ends at s_j-1)
        tt = target.square().sum()
        S = {kind: (sim(q, K, kind).double(), sim(yq, ys, kind).double()) for kind in SIMS}
        for (kind, te, ty) in combos:
            se, sy = S[kind]
            att = torch.softmax(te * se + ty * sy, -1)
            eh = (att[..., None] * V).sum(1)
            acc[kind, te, ty] += torch.stack([tt, (target * eh).sum(), eh.square().sum(),
                                              tt.new_tensor(float(target.numel()))])
    return {c: a.cpu() for c, a in acc.items()}


def mse_at(a, s):
    return float((a[0] - 2 * s * a[1] + s * s * a[2]) / a[3])


def families(kind):
    return {f'{kind}:mean': lambda te, ty: te == 0 and ty == 0,
            f'{kind}:e': lambda te, ty: te > 0 and ty == 0,
            f'{kind}:y': lambda te, ty: te == 0 and ty > 0,
            f'{kind}:e+y': lambda te, ty: te > 0 and ty > 0}


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    device = exp.device
    cell = CELLS[code]
    ages = demo_ages('phase', cell['history'], cell['period'], cell['stride'])
    val = load_split(exp, 'val', cell['history'], args.forecast_bs)
    test = load_split(exp, 'test', cell['history'], args.forecast_bs)
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    test_base = float(test['E'][test['origins'] - L].double().square().mean())
    if abs(test_base - official) > 1e-4:
        raise AssertionError(f'base cached={test_base} official={official}')
    del exp
    torch.cuda.empty_cache()
    grid = [(kind, te, ty) for kind in SIMS for te in TAUS for ty in TAUS]
    acc_v = sums(val, ages, device, args.bs, grid)
    chosen = {}
    for kind in SIMS:
        for f, ok in families(kind).items():
            best = min(((te, ty, s) for (kd, te, ty) in grid if kd == kind and ok(te, ty) for s in S_GRID),
                       key=lambda x: mse_at(acc_v[kind, x[0], x[1]], x[2]))
            chosen[f] = dict(kind=kind, tau_e=best[0], tau_y=best[1], s=best[2],
                             val_mse=mse_at(acc_v[kind, best[0], best[1]], best[2]))
    val_base = float(acc_v[grid[0]][0] / acc_v[grid[0]][3])
    acc_t = sums(test, ages, device, args.bs, sorted({(x['kind'], x['tau_e'], x['tau_y']) for x in chosen.values()}))
    any_a = next(iter(acc_t.values()))
    base = float(any_a[0] / any_a[3])
    assert abs(base - test_base) < 1e-6, (base, test_base)
    result = {}
    for f, x in chosen.items():
        m = mse_at(acc_t[x['kind'], x['tau_e'], x['tau_y']], x['s'])
        result[f] = dict(x, test_mse=m, test_gain_pct=100 * (base - m) / base)
        print(f'[test] {f:9s} tau_e={x["tau_e"]:<5} tau_y={x["tau_y"]:<5} s={x["s"]:<5} '
              f'mse={m:.6f} gain={result[f]["test_gain_pct"]:+.3f}%', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, W=W, ages=ages, taus=TAUS, s_grid=S_GRID, checkpoint=str(checkpoint),
               val_base=val_base, test_base=base, families=result)
    output = Path(args.out_dir) / f'{code}_H{H}_phase_rawsim_s{seed}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T', choices=CELLS)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--bs', type=int, default=2048)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
