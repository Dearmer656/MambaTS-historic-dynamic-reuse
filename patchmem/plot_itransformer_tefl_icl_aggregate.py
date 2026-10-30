"""Plot paper-protocol TEFL ICL results across seeds, with per-seed CSVs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='patchmem/logs/server/itr_tefl_icl')
    parser.add_argument('--output_dir', default='patchmem/logs/server/itr_tefl_icl/paper_figures')
    parser.add_argument('--datasets', default='E,W,T')
    parser.add_argument('--seeds', default='3047,2024,2025')
    parser.add_argument('--baseline_tag', default='TEFL_paper_L96_E192')
    args = parser.parse_args()
    src, dst = Path(args.input_dir), Path(args.output_dir)
    codes = args.datasets.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    results = defaultdict(list)
    for code in codes:
        for seed in seeds:
            path = src / f'{code}_{args.baseline_tag}_s{seed}.json'
            if path.exists():
                result = json.loads(path.read_text())
                assert result['seed'] == seed and result['baseline_tag'] == args.baseline_tag
                results[code].append(result)
    if not results:
        raise FileNotFoundError(f'No completed {args.baseline_tag} evaluations in {src}')
    dst.mkdir(parents=True, exist_ok=True)

    rows = []
    grouped = defaultdict(list)
    for code, replicates in results.items():
        for result in replicates:
            for cell in result['test'].values():
                for family, item in cell['arms'].items():
                    row = {'dataset': result['name'], 'code': code, 'seed': result['seed'],
                           'history': cell['N'], 'stride': cell['stride'],
                           'demos': cell['n_demos'], 'family': family,
                           'a1': item['a1'], 'a2': item['a2'],
                           'tefl_mse': cell['tefl_mse'], 'updated_mse': item['mse'],
                           'gain_pct': item['gain_pct']}
                    rows.append(row)
                    grouped[code, cell['N'], cell['stride'], family].append(row)
    with (dst / 'results_by_seed.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for (code, history, stride, family), group in sorted(grouped.items()):
        summary.append({'dataset': group[0]['dataset'], 'code': code,
                        'history': history, 'stride': stride, 'family': family,
                        'seeds': len(group),
                        'tefl_mse_mean': float(np.mean([r['tefl_mse'] for r in group])),
                        'updated_mse_mean': float(np.mean([r['updated_mse'] for r in group])),
                        'updated_mse_sd': float(np.std([r['updated_mse'] for r in group])),
                        'gain_pct_mean': float(np.mean([r['gain_pct'] for r in group])),
                        'gain_pct_sd': float(np.std([r['gain_pct'] for r in group]))})
    with (dst / 'results_mean_sd.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    plt.rcParams.update({'font.size': 10, 'font.family': 'serif',
                         'font.serif': ['DejaVu Serif'], 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 300})
    palette = {1440: '#0072B2', 2880: '#D55E00'}
    for metric, ylabel, filename in (
            ('gain_pct', 'Test MSE improvement over TEFL (%)', 'gain_by_stride_history'),
            ('updated_mse', 'Official test MSE', 'mse_by_stride_history')):
        fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.5), squeeze=False)
        for ax, (code, replicates) in zip(axes[0], results.items()):
            histories = sorted({k[1] for k in grouped if k[0] == code})
            if metric == 'updated_mse':
                baseline = np.mean([next(iter(r['test'].values()))['tefl_mse'] for r in replicates])
                ax.axhline(baseline, color='0.4', linestyle='--', label='TEFL')
            else:
                ax.axhline(0, color='0.4', linestyle='--')
            for history in histories:
                cells = [s for s in summary if s['code'] == code and
                         s['history'] == history and s['family'] == 'W1+W2']
                cells.sort(key=lambda s: s['stride'])
                x = np.array([s['stride'] for s in cells])
                y = np.array([s[f'{metric}_mean'] for s in cells])
                sd = np.array([s[f'{metric}_sd'] for s in cells])
                color = palette.get(history)
                ax.plot(x, y, marker='o', color=color, linewidth=1.8,
                        label=f'History {history} (n={cells[0]["seeds"]})')
                if len(replicates) > 1:
                    ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.16)
            ax.set_title(replicates[0]['name'])
            ax.set_xlabel('Demo stride (time steps)')
            ax.set_xticks(x)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.2)
            ax.legend(frameon=False, fontsize=8)
        axes[0][0].set_ylabel(ylabel)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(dst / f'{filename}.{ext}', bbox_inches='tight', dpi=300)
        plt.close(fig)
    print(dst)


if __name__ == '__main__':
    main()
