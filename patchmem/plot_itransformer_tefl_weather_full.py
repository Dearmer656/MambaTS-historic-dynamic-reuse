"""Compare all eligible error demonstrations with same-clock daily ones."""
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path('patchmem/logs/server/itr_tefl_icl')
OUT = ROOT / 'weather_full_lookback_figures'
SEEDS = (3047, 2024, 2025)
HISTORIES = (576, 720, 1008, 1440, 2880)
ARMS = ('W1+W2', 'W2_only', 'W1_only')


def main():
    rows = []
    for seed in SEEDS:
        full = json.loads((ROOT / f'W_full_lookback_s{seed}.json').read_text())
        periodic_files = [ROOT / f'W_short_history_s{seed}.json',
                          ROOT / f'W_periodic_s{seed}.json']
        periodic = {}
        for path in periodic_files:
            periodic.update(json.loads(path.read_text())['test'])
        for n in HISTORIES:
            for selection, x, val in (
                ('full', full['test'][f'N{n}_D1'], full['validation'][f'N{n}_D1']),
                ('daily', periodic[f'N{n}_P144'], None),
            ):
                for arm in ARMS:
                    a = x['arms'][arm]
                    accepted = (val['chosen'][arm]['mse'] < val['tefl_mse']
                                if val else a['val_prefers_update'])
                    rows.append(dict(seed=seed,history=n,selection=selection,arm=arm,
                                     demos=x['n_demos'],baseline_mse=x['tefl_mse'],
                                     mse=a['mse'],gain_pct=a['gain_pct'],
                                     val_prefers_update=accepted,
                                     guarded_gain_pct=a['gain_pct'] if accepted else 0.0))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'per_seed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for col, arm in enumerate(('W1+W2', 'W2_only')):
        for row in range(2):
            ax = axes[row, col]
            histories = HISTORIES if row == 0 else HISTORIES[2:]
            for selection, color in (('full', '#225ea8'), ('daily', '#d95f0e')):
                means, sds = [], []
                for n in histories:
                    v = [r['gain_pct'] for r in rows if r['history'] == n
                         and r['selection'] == selection and r['arm'] == arm]
                    means.append(statistics.mean(v))
                    sds.append(statistics.stdev(v))
                ax.errorbar(range(len(histories)), means, yerr=sds, marker='o',
                            capsize=3, label=selection, color=color)
            ax.axhline(0, color='black', lw=.8)
            ax.set_xticks(range(len(histories)), histories)
            ax.set_xlabel('History length N')
            ax.set_title(f'{arm}: {"all lengths" if row == 0 else "longer histories"}')
            ax.grid(alpha=.25)
            if row == 1:
                ax.set_ylim(-.8, .6)
    axes[0, 0].set_ylabel('Test MSE improvement over TEFL (%)')
    axes[1, 0].set_ylabel('Test MSE improvement over TEFL (%)')
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Weather, L=96, H=L_error=192; validation-selected steps; 3 seeds')
    fig.tight_layout()
    fig.savefig(OUT / 'full_vs_daily.png', dpi=180)
    fig.savefig(OUT / 'full_vs_daily.pdf')
    print(OUT / 'full_vs_daily.png')


if __name__ == '__main__':
    main()
