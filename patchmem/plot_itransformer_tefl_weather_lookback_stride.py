"""Weather lookback/stride plots from three paper-protocol TEFL seeds."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path('patchmem/logs/server/itr_tefl_icl')
OUT = ROOT / 'weather_lookback_stride_figures'
SEEDS = (3047, 2024, 2025)
LONG_HISTORIES = (1440, 2880)
ALL_HISTORIES = (576, 720, 1008, 1440, 2880)
STRIDES = (1, 48, 96, 144, 192, 288, 384)


def read_results():
    rows = []
    for seed in SEEDS:
        fixed = json.loads((ROOT / f'W_TEFL_paper_L96_E192_s{seed}.json').read_text())
        full = json.loads((ROOT / f'W_full_lookback_s{seed}.json').read_text())
        for n in ALL_HISTORIES:
            strides = STRIDES if n in LONG_HISTORIES else (1,)
            for stride in strides:
                cell = (full if stride == 1 else fixed)['test'][f'N{n}_D{stride}']
                arm = cell['arms']['W1+W2']
                rows.append(dict(seed=seed,history=n,stride=stride,demos=cell['n_demos'],
                                 baseline_mse=cell['tefl_mse'],mse=arm['mse'],
                                 gain_pct=arm['gain_pct'],selection='fixed'))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'per_seed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary = []
    for n in ALL_HISTORIES:
        for stride in (STRIDES if n in LONG_HISTORIES else (1,)):
            group = [r for r in rows if r['history'] == n and r['stride'] == stride]
            summary.append(dict(history=n,stride=stride,demos=group[0]['demos'],seeds=len(group),
                                baseline_mse_mean=mean(r['baseline_mse'] for r in group),
                                mse_mean=mean(r['mse'] for r in group),
                                mse_sd=stdev(r['mse'] for r in group),
                                gain_pct_mean=mean(r['gain_pct'] for r in group),
                                gain_pct_sd=stdev(r['gain_pct'] for r in group)))
    with (OUT / 'mean_sd.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    return summary


def heatmap(summary):
    gain = np.array([[next(r['gain_pct_mean'] for r in summary if r['history'] == n and
                           r['stride'] == d) for d in STRIDES] for n in LONG_HISTORIES])
    mse = np.array([[next(r['mse_mean'] for r in summary if r['history'] == n and
                          r['stride'] == d) for d in STRIDES] for n in LONG_HISTORIES])
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.5), constrained_layout=True)
    for ax, values, cmap, norm, colorbar in (
        (axes[0], gain, 'RdBu', TwoSlopeNorm(vmin=-.45, vcenter=0, vmax=.45), 'MSE improvement over TEFL (%)'),
        (axes[1], mse, 'viridis_r', None, 'Test MSE'),
    ):
        image = ax.imshow(values, aspect='auto', cmap=cmap, norm=norm)
        ax.set_xticks(range(len(STRIDES)), STRIDES)
        ax.set_yticks(range(len(LONG_HISTORIES)), LONG_HISTORIES)
        ax.set_xlabel('Demonstration stride (time steps)')
        ax.set_ylabel('Lookback length N')
        for i in range(len(LONG_HISTORIES)):
            for j in range(len(STRIDES)):
                v = values[i, j]
                label = f'{v:+.2f}%' if ax is axes[0] else f'{v:.4f}'
                ax.text(j, i, label, ha='center', va='center', fontsize=8,
                        color='white' if (ax is axes[0] and v < -.28) else 'black')
        fig.colorbar(image, ax=ax, label=colorbar, shrink=.85, pad=.02)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'lookback_stride_heatmap.{ext}', dpi=300)
    plt.close(fig)


def history_sweep(summary):
    daily = {}
    for seed in SEEDS:
        sources = [ROOT / f'W_short_history_s{seed}.json', ROOT / f'W_periodic_s{seed}.json']
        test = {}
        for source in sources:
            test.update(json.loads(source.read_text())['test'])
        for n in ALL_HISTORIES:
            daily.setdefault(n, []).append(test[f'N{n}_P144']['arms']['W1+W2']['gain_pct'])
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    full = [next(r for r in summary if r['history'] == n and r['stride'] == 1)
            for n in ALL_HISTORIES]
    positions = range(len(ALL_HISTORIES))
    ax.errorbar(positions, [r['gain_pct_mean'] for r in full],
                yerr=[r['gain_pct_sd'] for r in full], color='#0072B2',
                marker='o', capsize=3, label='Full lookback (stride 1)')
    ax.errorbar(positions, [mean(daily[n]) for n in ALL_HISTORIES],
                yerr=[stdev(daily[n]) for n in ALL_HISTORIES], color='#D55E00',
                marker='s', capsize=3, label='Same-clock daily demos')
    ax.axhline(0, color='0.3', linewidth=.8)
    ax.set_xticks(list(positions), ALL_HISTORIES)
    ax.set_xlabel('Lookback length N')
    ax.set_ylabel('Test MSE improvement over TEFL (%)')
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'history_sweep.{ext}', dpi=300)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif',
                         'font.serif': ['DejaVu Serif'], 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    summary = read_results()
    heatmap(summary)
    history_sweep(summary)
    print(OUT)


if __name__ == '__main__':
    main()
