"""Plot Weather daily/weekly history sweeps and Traffic's 14-demo reference."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path('patchmem/logs/server/itr_tefl_icl')
OUT = ROOT / 'weather_long_periodic_figures'
SEEDS = (3047, 2024, 2025)
WEATHER = ((1440, 'W_periodic_s{}.json'),
           (2448, 'W_daily_match_s{}.json'),
           (2880, 'W_periodic_s{}.json'),
           (4320, 'W_long_periodic_s{}.json'),
           (5760, 'W_long_periodic_s{}.json'),
           (10080, 'W_long_periodic_s{}.json'),
           (14400, 'W_14_weekly_s{}.json'))


def collect():
    rows = []
    for seed in SEEDS:
        for n, pattern in WEATHER:
            test = json.loads((ROOT / pattern.format(seed)).read_text())['test']
            for period, name in ((144, 'daily'), (1008, 'weekly')):
                key = f'N{n}_P{period}'
                if key not in test:
                    continue
                x = test[key]
                arm = x['arms']['W1+W2']
                rows.append(dict(dataset='Weather',seed=seed,history=n,period=name,
                                 demos=x['n_demos'],baseline_mse=x['tefl_mse'],
                                 mse=arm['mse'],gain_pct=arm['gain_pct'],
                                 guarded_gain_pct=arm['guarded_gain_pct'],
                                 val_prefers_update=arm['val_prefers_update']))
        x = json.loads((ROOT / f'W_daily_match_s{seed}.json').read_text())['test']['N14400_P144']
        arm = x['arms']['W1+W2']
        rows.append(dict(dataset='Weather',seed=seed,history=14400,period='daily',
                         demos=x['n_demos'],baseline_mse=x['tefl_mse'],mse=arm['mse'],
                         gain_pct=arm['gain_pct'],guarded_gain_pct=arm['guarded_gain_pct'],
                         val_prefers_update=arm['val_prefers_update']))
        x = json.loads((ROOT / f'T_periodic_s{seed}.json').read_text())['test']['N2880_P168']
        arm = x['arms']['W1+W2']
        rows.append(dict(dataset='Traffic',seed=seed,history=2880,period='weekly',
                         demos=x['n_demos'],baseline_mse=x['tefl_mse'],
                         mse=arm['mse'],gain_pct=arm['gain_pct'],
                         guarded_gain_pct=arm['guarded_gain_pct'],
                         val_prefers_update=arm['val_prefers_update']))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'per_seed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary = []
    for dataset, period, history in sorted({(r['dataset'],r['period'],r['history']) for r in rows}):
        group = [r for r in rows if (r['dataset'],r['period'],r['history']) ==
                 (dataset,period,history)]
        summary.append(dict(dataset=dataset,period=period,history=history,
                            demos=group[0]['demos'],seeds=len(group),
                            baseline_mse_mean=mean(r['baseline_mse'] for r in group),
                            mse_mean=mean(r['mse'] for r in group),
                            gain_pct_mean=mean(r['gain_pct'] for r in group),
                            gain_pct_sd=stdev(r['gain_pct'] for r in group),
                            guarded_gain_pct_mean=mean(r['guarded_gain_pct'] for r in group),
                            val_accept_count=sum(r['val_prefers_update'] for r in group)))
    with (OUT / 'mean_sd.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    return summary


def make_plot(summary):
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif',
                         'font.serif': ['DejaVu Serif'], 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    colors = {'daily': '#0072B2', 'weekly': '#D55E00'}
    history_positions = {n:i for i,(n,_) in enumerate(WEATHER)}
    for period in ('daily', 'weekly'):
        group = sorted((r for r in summary if r['dataset'] == 'Weather' and
                        r['period'] == period), key=lambda r: r['history'])
        for ax, field in ((axes[0], 'history'), (axes[1], 'demos')):
            x = [history_positions[r['history']] if field == 'history' else r['demos'] for r in group]
            ax.errorbar(x, [r['gain_pct_mean'] for r in group],
                        yerr=[r['gain_pct_sd'] for r in group], marker='o',
                        capsize=3, color=colors[period], label=f'Weather {period}')
    traffic = next(r for r in summary if r['dataset'] == 'Traffic')
    axes[1].errorbar([traffic['demos']], [traffic['gain_pct_mean']],
                     yerr=[traffic['gain_pct_sd']], marker='s', capsize=3,
                     color='#009E73', label='Traffic weekly')
    axes[0].set_xticks(range(len(WEATHER)), [n for n,_ in WEATHER])
    axes[0].tick_params(axis='x', rotation=35)
    axes[0].set_xlabel('Weather lookback length N')
    axes[1].set_xlabel('Same-clock demonstrations')
    axes[1].set_xticks([1,14,27,37,67,97])
    for ax in axes:
        ax.axhline(0, color='0.3', linewidth=.8)
        ax.grid(axis='y', alpha=.2)
        ax.set_ylabel('Test MSE improvement over TEFL (%)')
        ax.legend(frameon=False, fontsize=8)
    axes[1].set_xlim(0,100)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'gain_by_history_and_demo_count.{ext}', dpi=300)
    plt.close(fig)


def main():
    summary = collect()
    make_plot(summary)
    print(OUT)


if __name__ == '__main__':
    main()
