"""Compare fixed-stride and same-phase TEFL in-context updates."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


SEEDS = {'W': (3047, 2024, 2025), 'T': (3047, 2024, 2025)}
PERIODS = {'W': (144, 1008), 'T': (24, 168)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='patchmem/logs/server/itr_tefl_icl')
    parser.add_argument('--output_dir', default='patchmem/logs/server/itr_tefl_icl/periodic_figures')
    parser.add_argument('--datasets', default='W,T')
    args = parser.parse_args()
    src, dst = Path(args.input_dir), Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    raw = []
    summary = []
    for code in args.datasets.split(','):
        loaded = []
        for seed in SEEDS[code]:
            fixed_path = src / f'{code}_TEFL_paper_L96_E192_s{seed}.json'
            phase_path = src / f'{code}_periodic_s{seed}.json'
            if not fixed_path.exists() or not phase_path.exists():
                continue
            fixed, phase = json.loads(fixed_path.read_text()), json.loads(phase_path.read_text())
            assert fixed['seed'] == phase['seed'] == seed
            assert fixed['checkpoint'] == phase['checkpoint']
            loaded.append((fixed, phase))
            for key, cell in fixed['test'].items():
                a = cell['arms']['W1+W2']
                raw.append({'dataset': fixed['name'], 'seed': seed, 'history': cell['N'],
                            'kind': 'fixed', 'setting': f'stride {cell["stride"]}',
                            'n_demos': cell['n_demos'], 'tefl_mse': cell['tefl_mse'],
                            'updated_mse': a['mse'], 'gain_pct': a['gain_pct'],
                            'guarded_gain_pct': ''})
            for key, cell in phase['test'].items():
                a = cell['arms']['W1+W2']
                match = fixed['test'][f'N{cell["history"]}_D48']
                assert abs(match['tefl_mse'] - cell['tefl_mse']) < 1e-4
                raw.append({'dataset': fixed['name'], 'seed': seed,
                            'history': cell['history'], 'kind': 'periodic',
                            'setting': f'period {cell["period"]}',
                            'n_demos': cell['n_demos'], 'tefl_mse': cell['tefl_mse'],
                            'updated_mse': a['mse'], 'gain_pct': a['gain_pct'],
                            'guarded_gain_pct': a['guarded_gain_pct']})
        if not loaded:
            continue
        group = defaultdict(list)
        for row in raw:
            if row['dataset'] == loaded[0][0]['name']:
                group[row['history'], row['kind'], row['setting']].append(row)
        for (history, kind, setting), reps in group.items():
            summary.append({'dataset': reps[0]['dataset'], 'history': history,
                            'kind': kind, 'setting': setting, 'seeds': len(reps),
                            'n_demos': reps[0]['n_demos'],
                            'tefl_mse_mean': float(np.mean([r['tefl_mse'] for r in reps])),
                            'updated_mse_mean': float(np.mean([r['updated_mse'] for r in reps])),
                            'gain_pct_mean': float(np.mean([r['gain_pct'] for r in reps])),
                            'gain_pct_sd': float(np.std([r['gain_pct'] for r in reps])),
                            'guarded_gain_pct_mean': (
                                float(np.mean([r['guarded_gain_pct'] for r in reps]))
                                if kind == 'periodic' else '')})
        histories = sorted({c['N'] for c in loaded[0][0]['test'].values()})
        fig, axes = plt.subplots(1, len(histories), figsize=(5.8 * len(histories), 3.6), squeeze=False)
        for ax, history in zip(axes[0], histories):
            items = [s for s in summary if s['dataset'] == loaded[0][0]['name']
                     and s['history'] == history]
            fixed_items = sorted((s for s in items if s['kind'] == 'fixed'),
                                 key=lambda s: int(s['setting'].split()[-1]))
            phase_items = sorted((s for s in items if s['kind'] == 'periodic'),
                                 key=lambda s: int(s['setting'].split()[-1]))
            ordered = fixed_items + phase_items
            x = np.arange(len(ordered))
            y = [s['gain_pct_mean'] for s in ordered]
            sd = [s['gain_pct_sd'] for s in ordered]
            colors = ['#0072B2'] * len(fixed_items) + ['#D55E00', '#009E73'][:len(phase_items)]
            ax.bar(x, y, color=colors, width=0.72)
            if len(loaded) > 1:
                ax.errorbar(x, y, yerr=sd, fmt='none', color='0.2', capsize=2, linewidth=0.8)
            ax.axhline(0, color='0.35', linewidth=0.9)
            ax.set_xticks(x, [s['setting'].replace('stride ', 'D')
                               .replace('period ', 'P') for s in ordered], rotation=45)
            ax.set_title(f'{loaded[0][0]["name"]}: history {history} (n={len(loaded)} seeds)')
            ax.set_ylabel('Test MSE improvement over TEFL (%)')
            ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(dst / f'{code}_fixed_vs_periodic.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
    if not raw:
        raise FileNotFoundError('No matched fixed and periodic results')
    for name, rows in (('results_by_seed.csv', raw), ('comparison_mean_sd.csv', summary)):
        with (dst / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(dst)


if __name__ == '__main__':
    main()
