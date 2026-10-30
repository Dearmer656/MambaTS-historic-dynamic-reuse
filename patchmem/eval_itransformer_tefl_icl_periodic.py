"""Same-clock-time TEFL W1/W2 updates with causal periodic demonstrations.

For a query forecast at origin t, a demo forecast at s is eligible only when
t-s is a multiple of the chosen period P and s+H <= t.  Its TEFL residual
E[s]-g(E[s-H]) is used by the existing in-context update.  Each setting's
step sizes are chosen on validation, then evaluated once on test.
"""
import argparse
import json
import os
from pathlib import Path

import torch

from patchmem.eval_itransformer_tefl_icl import (
    A1_GRID, A2_GRID, CONFIG, FAMILIES, forecast_errors, load_model, score_grid,
    score_tefl,
)


DEFAULT_PERIODS = {
    'E': (24, 168), 'M1': (96, 672), 'M2': (96, 672),
    'W': (144, 1008), 'T': (24, 168),
}
L = 96
H = int(os.environ.get('ITR_H', '192'))  # forecast horizon; set ITR_H for H != 192


def aligned_demo_ages(history, period):
    """Ages t-s of fully observed demos whose keys fit inside history."""
    assert history >= L + 2 * H and period > 0
    first = ((H + period - 1) // period) * period
    last = history - H - L
    return list(range(first, last + 1, period))


@torch.inference_mode()
def run(code, seed, histories, periods, chunk, forecast_bs, output, baseline_tag):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, checkpoint = load_model(code, seed, device, baseline_tag)
    W1 = exp.model.W1.weight.detach().to(device=device, dtype=torch.float64)
    W2 = exp.model.W2.weight.detach().to(device=device, dtype=torch.float64)
    candidates = [(a1, a2) for a1 in A1_GRID for a2 in A2_GRID if a1 or a2]
    cells = [(N, P, aligned_demo_ages(N, P)) for N in histories for P in periods]
    if any(not ages for _, _, ages in cells):
        raise ValueError(f'At least one period has no eligible demos: {cells}')
    validation, test, counts = {}, {}, {}

    for split in ('val', 'test'):
        E, origins, channels = forecast_errors(exp, split, max(histories), forecast_bs)
        counts[split] = len(origins)
        baseline = score_tefl(E, origins, W1, W2, chunk)
        print(f'[{split}] cases={len(origins)} channels={channels} TEFL={baseline:.8f}', flush=True)
        if split == 'test':
            metrics = checkpoint.parents[1].parent / 'results' / checkpoint.parent.name / 'tefl_metrics.json'
            official = json.loads(metrics.read_text())['mse']
            if abs(baseline - official) > 1e-4:
                raise AssertionError(f'TEFL cached={baseline} official={official}')
        for N, P, ages in cells:
            key = f'N{N}_P{P}'
            if split == 'val':
                scores, n = score_grid(E, origins, W1, W2, N, P, chunk, candidates, ages)
                chosen = {}
                for family, admissible in FAMILIES.items():
                    valid = [pair for pair in candidates if admissible(*pair)]
                    best = min(valid, key=lambda pair: float(scores[pair].mean()))
                    mse = float(scores[best].mean())
                    chosen[family] = {'a1': best[0], 'a2': best[1], 'mse': mse,
                                      'val_prefers_update': mse < baseline}
                validation[key] = {'history': N, 'period': P, 'n_demos': n,
                                   'demo_ages': ages, 'tefl_mse': baseline, 'chosen': chosen}
                print(f'[val] {key} demos={n} TEFL={baseline:.6f} '
                      f'W1+W2={chosen["W1+W2"]}', flush=True)
            else:
                selected = [(x['a1'], x['a2']) for x in validation[key]['chosen'].values()]
                scores, n = score_grid(E, origins, W1, W2, N, P, chunk, selected, ages)
                arms = {}
                for family, picked in validation[key]['chosen'].items():
                    mse = float(scores[picked['a1'], picked['a2']].mean())
                    guarded = mse if picked['val_prefers_update'] else baseline
                    arms[family] = {**picked, 'mse': mse,
                                    'gain_pct': 100 * (baseline - mse) / baseline,
                                    'guarded_mse': guarded,
                                    'guarded_gain_pct': 100 * (baseline - guarded) / baseline}
                test[key] = {'history': N, 'period': P, 'n_demos': n,
                             'tefl_mse': baseline, 'arms': arms}
                print(f'[test] {key} TEFL={baseline:.6f} '
                      f'W1+W2={arms["W1+W2"]["mse"]:.6f} '
                      f'gain={arms["W1+W2"]["gain_pct"]:+.2f}% '
                      f'guarded={arms["W1+W2"]["guarded_gain_pct"]:+.2f}%', flush=True)
        del E
    result = {'dataset': code, 'name': CONFIG[code]['name'], 'seed': seed,
              'model': 'iTransformer-TEFL', 'L': L, 'H': H, 'L_error': H,
              'checkpoint': str(checkpoint), 'baseline_tag': baseline_tag,
              'selection': 'same-period phase; fully observed demos only',
              'validation_windows': counts['val'], 'test_windows': counts['test'],
              'validation': validation, 'test': test}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, choices=DEFAULT_PERIODS)
    parser.add_argument('--seed', type=int, default=3047)
    parser.add_argument('--histories', default='1440,2880')
    parser.add_argument('--periods', default=None)
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--forecast_bs', type=int, default=16)
    parser.add_argument('--baseline_tag', default='TEFL_paper_L96_E192')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    histories = [int(x) for x in args.histories.split(',')]
    periods = [int(x) for x in args.periods.split(',')] if args.periods else DEFAULT_PERIODS[args.data]
    assert histories and periods and args.chunk > 0 and args.forecast_bs > 0
    output = Path(args.out or f'patchmem/logs/server/itr_tefl_icl/{args.data}_periodic_s{args.seed}.json')
    run(args.data, args.seed, histories, periods, args.chunk, args.forecast_bs,
        output, args.baseline_tag)


if __name__ == '__main__':
    main()
