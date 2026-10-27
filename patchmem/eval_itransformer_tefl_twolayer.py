"""Case-wise sequential GD on TEFL's own head: bias step, then W2, then W1.

TEFL head: g(eps) = W2 ReLU(W1 eps) (+ b, b = 0 after training).  For each
query, with same-phase demos (key eps_j, TEFL residual r_j = y_j - y_tefl_j):

  1. bias  : b  = a_b * mean_j r_j                 (GD step on b, step a_b/n)
  2. W2    : v_j = r_j - b  (residual after the bias step, = y - y_tefl - y_avg)
             dW2 = eta * sum_j v_j h_j^T,  eta = a2 / (|h_q| sum_j |h_j|)
  3. W1    : e_j = v_j - dW2 h_j  (residual after steps 1-2)
             dW1 = sum_j [((W2 + dW2)^T e_j) * 1(W1 k_j > 0)] k_j^T / sum_j |k_j|^2
  pred = (W2 + dW2) ReLU((W1 + a1 dW1) q) + b

a_b = a2 = a1 = 0 is TEFL.  Special cases match earlier evaluators: a2 = a1 = 0
is the phase mean, a_b = a1 = 0 the W2-only step, a_b = a2 = 0 the W1-only step.
Step sizes are chosen per family on the official val split; test is scored once.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_itransformer_tefl_icl import forecast_errors, load_model
from patchmem.eval_itransformer_tefl_avg_icl import PERIODS, history_for, phase_ages

L, H = 96, 192
AB_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
A2_GRID = (0.0, 0.1, 0.3, 1.0, 3.0)
A1_GRID = (0.0, 0.03, 0.1, 0.3, 1.0)
FAMILIES = {
    'avg': lambda ab, a2, a1: ab > 0 and a2 == 0 and a1 == 0,
    'avg+W2': lambda ab, a2, a1: ab > 0 and a2 > 0 and a1 == 0,
    'avg+W2+W1': lambda ab, a2, a1: ab > 0 and a2 > 0 and a1 > 0,
    'W2': lambda ab, a2, a1: ab == 0 and a2 > 0 and a1 == 0,
    'W2+W1': lambda ab, a2, a1: ab == 0 and a2 > 0 and a1 > 0,
}


@torch.inference_mode()
def score_chunk(q, K, V, target, W1, W2, combos):
    """Per-case MSE for each (a_b, a2, a1); low-rank dW2 kept implicit via Gram products."""
    pre_q, pre_k = q @ W1.T, K @ W1.T               # b C R / b C n R
    hq, hk = torch.relu(pre_q), torch.relu(pre_k)
    mask = (pre_k > 0).to(K.dtype)
    r = V - hk @ W2.T                               # b C n H
    r_bar = r.mean(-2)
    G = hk @ hk.transpose(-1, -2)                   # b C n n
    kq = (K * q[:, :, None]).sum(-1)                # b C n
    ksq = K.square().sum((-1, -2)).clamp_min(1e-12)  # b C
    den = hq.norm(dim=-1) * hk.norm(dim=-1).sum(-1) + 1e-6
    out = {}
    for ab in sorted({c[0] for c in combos}):
        b = ab * r_bar
        v = r - b[:, :, None]
        for a2 in sorted({c[1] for c in combos if c[0] == ab}):
            eta = (a2 / den)[..., None, None]
            e = v - eta * (G @ v)                   # dW2 h_j = eta sum_i <h_i, h_j> v_i
            eW2 = e @ W2 + eta * ((e @ v.transpose(-1, -2)) @ hk)
            dq = ((eW2 * mask) * kq[..., None]).sum(-2) / ksq[..., None]
            for a1 in sorted({c[2] for c in combos if c[:2] == (ab, a2)}):
                hq1 = torch.relu(pre_q + a1 * dq)
                sim = (hk * hq1[:, :, None]).sum(-1)
                pred = hq1 @ W2.T + eta[..., 0] * (sim[..., None] * v).sum(-2) + b
                out[ab, a2, a1] = (target - pred).square().mean((1, 2)).cpu()
    return out


@torch.inference_mode()
def score_cell(E, origins, W1, W2, ages, chunk, combos):
    ages = torch.as_tensor(ages, dtype=torch.long)
    dev = W1.device
    res = {c: torch.empty(len(origins), dtype=torch.float64) for c in combos}
    for start in range(0, len(origins), chunk):
        t = origins[start:start + chunk]
        s = t[:, None] - ages[None]
        q = E[t - H - L].permute(0, 2, 1).to(device=dev, dtype=W1.dtype)
        target = E[t - L].permute(0, 2, 1).to(device=dev, dtype=W1.dtype)
        K = E[s - H - L].permute(0, 3, 1, 2).to(device=dev, dtype=W1.dtype)
        V = E[s - L].permute(0, 3, 1, 2).to(device=dev, dtype=W1.dtype)
        for c, m in score_chunk(q, K, V, target, W1, W2, combos).items():
            res[c][start:start + len(t)] = m
    return res


def run(code, seed, ns, chunk, forecast_bs, output, baseline_tag):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, checkpoint = load_model(code, seed, device, baseline_tag)
    W1 = exp.model.W1.weight.detach().to(device=device, dtype=torch.float64)
    W2 = exp.model.W2.weight.detach().to(device=device, dtype=torch.float64)
    cells = {f'{ph}_n{n}': phase_ages(P, n) for ph, P in zip(('day', 'week'), PERIODS[code]) for n in ns}
    history = max(history_for(a) for a in cells.values())
    grid = [(ab, a2, a1) for ab in AB_GRID for a2 in A2_GRID for a1 in A1_GRID]
    chosen, tefl = {}, {}
    result = {}
    for split in ('val', 'test'):
        E, origins, channels = forecast_errors(exp, split, history, forecast_bs)
        for cell, ages in cells.items():
            combos = grid if split == 'val' else sorted({x['steps'] for x in chosen[cell].values()} | {(0.0, 0.0, 0.0)})
            scores = score_cell(E, origins, W1, W2, ages, chunk, combos)
            base = float(scores[0.0, 0.0, 0.0].mean())
            tefl[split] = base
            if split == 'val':
                chosen[cell] = {}
                for fam, ok in FAMILIES.items():
                    best = min((c for c in grid if ok(*c)), key=lambda c: float(scores[c].mean()))
                    chosen[cell][fam] = dict(steps=best, val_mse=float(scores[best].mean()))
            else:
                fams = {}
                for fam, x in chosen[cell].items():
                    m = float(scores[x['steps']].mean())
                    fams[fam] = dict(steps=x['steps'], val_mse=x['val_mse'], test_mse=m,
                                     test_gain_pct=100 * (base - m) / base)
                avg_case = scores[chosen[cell]['avg']['steps']]
                win = {f: float((scores[chosen[cell][f]['steps']] < avg_case).double().mean())
                       for f in ('avg+W2', 'avg+W2+W1')}
                result[cell] = dict(n_demos=len(ages), ages=ages, families=fams,
                                    case_win_rate_vs_avg=win)
                print(f'[test] {cell:9s} ' + ' '.join(
                    f'{f}={v["test_mse"]:.6f}({v["test_gain_pct"]:+.2f}%,{v["steps"]})' for f, v in fams.items()),
                    flush=True)
        print(f'[{split}] cases={len(origins)} channels={channels} history={history} TEFL={tefl[split]:.6f}', flush=True)
        if split == 'test':
            metrics = checkpoint.parents[1].parent / 'results' / checkpoint.parent.name / 'tefl_metrics.json'
            official = json.loads(metrics.read_text())['mse']
            if abs(tefl['test'] - official) > 1e-4:
                raise AssertionError(f'TEFL cached={tefl["test"]} official={official}')
        del E
    out = dict(dataset=code, seed=seed, L=L, H=H, L_error=H, periods=PERIODS[code], history=history,
               grids=dict(a_b=AB_GRID, a2=A2_GRID, a1=A1_GRID), checkpoint=str(checkpoint),
               baseline_tag=baseline_tag, val_tefl=tefl['val'], test_tefl=tefl['test'], cells=result)
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
    output = Path(args.out or f'patchmem/logs/server/itr_twolayer/{args.data}_s{args.seed}.json')
    run(args.data, args.seed, ns, args.chunk, args.forecast_bs, output, args.baseline_tag)


if __name__ == '__main__':
    main()
