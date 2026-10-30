"""TEFL + same-phase average + centered softmax ICL, with daily vs weekly demos.

Decomposition of the TEFL residual of a demo:  r_j = r_avg + (r_j - r_avg).
The average takes the first term; the in-context step only sees the centered
values v_j = r_j - r_avg, weighted by a softmax over centered key/query
similarities (keys and query minus the demo-key mean in h = ReLU(W1 eps)):

  avg            r_avg                                   (scale a = 1)
  avg_valA       a * r_avg, a chosen on val              (reference only)
  avg_icl_dot    r_avg + sum_j softmax(<k_j, q>/sqrt(R)/tau_dot) v_j
  avg_icl_cos    r_avg + sum_j softmax(cos(k_j, q)/tau_cos) v_j

Both phases (day, week) use the same number n of demos on every dataset, all
fully observed by the query origin (age >= H).  tau and a2 = 1 are fixed in
advance; nothing except avg_valA's scale touches validation.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_itransformer_tefl_icl import forecast_errors, load_model

L, H = 96, 192
PERIODS = {'E': (24, 168), 'M1': (96, 672), 'M2': (96, 672), 'W': (144, 1008), 'T': (24, 168)}
TAU_DOT, TAU_COS = 1.0, 0.1
A_GRID = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5)


def phase_ages(period, n):
    """n most recent same-phase ages whose targets are fully observed (age >= H)."""
    first = ((H + period - 1) // period) * period
    return [first + period * k for k in range(n)]


def history_for(ages):
    return max(ages) + H + L


@torch.inference_mode()
def corrections(q, K, V, W1, W2):
    hq = torch.relu(q @ W1.T)                       # b C R
    hk = torch.relu(K @ W1.T)                       # b C n R
    gq = hq @ W2.T
    r = V - hk @ W2.T                               # b C n H
    r_avg = r.mean(-2)
    v = r - r_avg[:, :, None]
    m = hk.mean(-2, keepdim=True)
    kc, qc = hk - m, hq - m[:, :, 0]
    dot = (kc * qc[:, :, None]).sum(-1) / math.sqrt(hk.shape[-1])
    cos = (kc * qc[:, :, None]).sum(-1) / (kc.norm(dim=-1) * qc.norm(dim=-1)[..., None] + 1e-12)
    out, diag = {'avg': r_avg}, {}
    for name, s, tau in (('dot', dot, TAU_DOT), ('cos', cos, TAU_COS)):
        w = torch.softmax(s / tau, dim=-1)
        out[f'icl_{name}'] = r_avg + (w[..., None] * v).sum(-2)
        diag[f'neff_frac_{name}'] = (1.0 / w.square().sum(-1)) / w.shape[-1]
    diag['dot_std'] = dot.std(-1)
    diag['cos_std'] = cos.std(-1)
    return gq, out, diag


@torch.inference_mode()
def collect(E, origins, W1, W2, ages, chunk):
    ages = torch.as_tensor(ages, dtype=torch.long)
    dev = W1.device
    S0 = torch.empty(len(origins), dtype=torch.float64)
    S1, S2, D = {}, {}, {}
    for start in range(0, len(origins), chunk):
        t = origins[start:start + chunk]
        sl = slice(start, start + len(t))
        sidx = t[:, None] - ages[None]
        q = E[t - H - L].permute(0, 2, 1).to(device=dev, dtype=W1.dtype)
        target = E[t - L].permute(0, 2, 1).to(device=dev, dtype=W1.dtype)
        K = E[sidx - H - L].permute(0, 3, 1, 2).to(device=dev, dtype=W1.dtype)
        V = E[sidx - L].permute(0, 3, 1, 2).to(device=dev, dtype=W1.dtype)
        gq, cs, diag = corrections(q, K, V, W1, W2)
        tr = target - gq
        S0[sl] = tr.square().mean((1, 2)).cpu()
        for k, c in cs.items():
            S1.setdefault(k, torch.empty(len(origins), dtype=torch.float64))[sl] = (tr * c).mean((1, 2)).cpu()
            S2.setdefault(k, torch.empty(len(origins), dtype=torch.float64))[sl] = c.square().mean((1, 2)).cpu()
        for k, v in diag.items():
            D.setdefault(k, torch.empty(len(origins), dtype=torch.float64))[sl] = v.mean(1).cpu()
    return S0, S1, S2, D


def mse(S0, S1, S2, k, a):
    return S0 - 2 * a * S1[k] + a * a * S2[k]


def run(code, seed, ns, chunk, forecast_bs, output, baseline_tag):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, checkpoint = load_model(code, seed, device, baseline_tag)
    W1 = exp.model.W1.weight.detach().to(device=device, dtype=torch.float64)
    W2 = exp.model.W2.weight.detach().to(device=device, dtype=torch.float64)
    cells = {f'{ph}_n{n}': phase_ages(P, n) for ph, P in zip(('day', 'week'), PERIODS[code]) for n in ns}
    history = max(history_for(a) for a in cells.values())
    per = {}
    for split in ('val', 'test'):
        E, origins, channels = forecast_errors(exp, split, history, forecast_bs)
        stats = {}
        for cell, ages in cells.items():
            stats[cell] = collect(E, origins, W1, W2, ages, chunk)
        tefl = float(next(iter(stats.values()))[0].mean())
        print(f'[{split}] cases={len(origins)} channels={channels} history={history} TEFL={tefl:.6f}', flush=True)
        if split == 'test':
            metrics = checkpoint.parents[1].parent / 'results' / checkpoint.parent.name / 'tefl_metrics.json'
            official = json.loads(metrics.read_text())['mse']
            if abs(tefl - official) > 1e-4:
                raise AssertionError(f'TEFL cached={tefl} official={official}')
        per[split] = (tefl, stats)
        del E
    val_tefl, val_stats = per['val']
    test_tefl, test_stats = per['test']
    result = {}
    for cell, ages in cells.items():
        vS0, vS1, vS2, _ = val_stats[cell]
        S0, S1, S2, D = test_stats[cell]
        a_val = min(A_GRID, key=lambda a: float(mse(vS0, vS1, vS2, 'avg', a).mean()))
        arms = {'avg': float(mse(S0, S1, S2, 'avg', 1.0).mean()),
                'avg_valA': float(mse(S0, S1, S2, 'avg', a_val).mean()),
                'avg_icl_dot': float(mse(S0, S1, S2, 'icl_dot', 1.0).mean()),
                'avg_icl_cos': float(mse(S0, S1, S2, 'icl_cos', 1.0).mean())}
        val_arms = {'avg': float(mse(vS0, vS1, vS2, 'avg', 1.0).mean()),
                    'avg_icl_dot': float(mse(vS0, vS1, vS2, 'icl_dot', 1.0).mean()),
                    'avg_icl_cos': float(mse(vS0, vS1, vS2, 'icl_cos', 1.0).mean())}
        avg_case = mse(S0, S1, S2, 'avg', 1.0)
        wins = {k: float((mse(S0, S1, S2, k[4:], 1.0) < avg_case).double().mean())
                for k in ('avg_icl_dot', 'avg_icl_cos')}
        result[cell] = dict(n_demos=len(ages), ages=ages, a_val=a_val, test=arms, val=val_arms,
                            test_gain_pct={k: 100 * (test_tefl - v) / test_tefl for k, v in arms.items()},
                            icl_vs_avg_case_win_rate=wins,
                            test_diag={k: float(v.mean()) for k, v in D.items()})
        g = result[cell]['test_gain_pct']
        print(f'[test] {cell:9s} n={len(ages)} avg={arms["avg"]:.6f} ({g["avg"]:+.2f}%) '
              f'avg_valA(a={a_val})={arms["avg_valA"]:.6f} ({g["avg_valA"]:+.2f}%) '
              f'icl_dot={arms["avg_icl_dot"]:.6f} ({g["avg_icl_dot"]:+.2f}%) '
              f'icl_cos={arms["avg_icl_cos"]:.6f} ({g["avg_icl_cos"]:+.2f}%)', flush=True)
    out = dict(dataset=code, seed=seed, L=L, H=H, L_error=H, periods=PERIODS[code], history=history,
               tau_dot=TAU_DOT, tau_cos=TAU_COS, a2=1.0, checkpoint=str(checkpoint),
               baseline_tag=baseline_tag, val_tefl=val_tefl, test_tefl=test_tefl, cells=result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, choices=PERIODS)
    parser.add_argument('--seed', type=int, default=3047)
    parser.add_argument('--ns', default='4,8,16')
    parser.add_argument('--chunk', type=int, default=4)
    parser.add_argument('--forecast_bs', type=int, default=8)
    parser.add_argument('--baseline_tag', default='TEFL_paper_L96_E192')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    ns = [int(x) for x in args.ns.split(',')]
    output = Path(args.out or f'patchmem/logs/server/itr_avg_icl/{args.data}_s{args.seed}.json')
    run(args.data, args.seed, ns, args.chunk, args.forecast_bs, output, args.baseline_tag)


if __name__ == '__main__':
    main()
