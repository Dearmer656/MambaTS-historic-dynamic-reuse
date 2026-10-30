"""One reproducible figure comparing ICL, TEFL and TEFL+ICL."""
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path('patchmem/logs/server')
OUT = ROOT / 'method_comparison'
SEEDS = (2024, 2025, 3047)
CONFIG = {
    'E': ('ETTh2', 'N1440_D48', '1440 / stride 48', 'E_TEFL_paper_L96_E192'),
    'M1': ('ETTm1', 'N2880_D96', '2880 / stride 96', 'M1_TEFL_paper_L96_E192'),
    'M2': ('ETTm2', 'N2880_D96', '2880 / stride 96', 'M2_TEFL_paper_L96_E192'),
    'W': ('Weather', 'N5760_P144', '5760 / same time daily', 'W_long_periodic'),
    'T': ('Traffic', 'N2880_P168', '2880 / same time weekly', 'T_periodic'),
}
METHODS = ('ICL only', 'TEFL only', 'TEFL + ICL')


def load_results():
    rows = []
    for code, (name, cell, rule, tag) in CONFIG.items():
        for seed in SEEDS:
            pure_path = ROOT / 'itr_pure_icl' / f'{code}_pure_icl_clip0p1_s{seed}.json'
            tefl_path = ROOT / 'itr_tefl_icl' / f'{tag}_s{seed}.json'
            pure, tefl = json.loads(pure_path.read_text()), json.loads(tefl_path.read_text())
            assert (pure['L'], pure['H'], pure['L_error']) == (96, 192, 192)
            assert (tefl['L'], tefl['H'], tefl['L_error']) == (96, 192, 192)
            assert pure['test_windows'] == tefl['test_windows']
            a, b = pure['test'][cell], tefl['test'][cell]
            assert a['n_demos'] == b['n_demos']
            rows.append(dict(dataset=name, cell=cell, history_rule=rule,
                             seed=seed, test_windows=pure['test_windows'],
                             n_demos=a['n_demos'],
                             itransformer_base_mse=a['base_mse'],
                             icl_only_mse=a['arms']['W1+W2']['mse'],
                             tefl_only_mse=b['tefl_mse'],
                             tefl_icl_mse=b['arms']['W1+W2']['mse']))
    return rows


def summarize(rows):
    summary = []
    for code, (name, cell, rule, _) in CONFIG.items():
        group = [r for r in rows if r['dataset'] == name]
        assert len(group) == len(SEEDS)
        item = dict(dataset=name, cell=cell, history_rule=rule,
                    seeds=len(group), n_demos=group[0]['n_demos'])
        for source, short in [('itransformer_base_mse', 'itransformer_base'),
                              ('icl_only_mse', 'icl_only'),
                              ('tefl_only_mse', 'tefl_only'),
                              ('tefl_icl_mse', 'tefl_icl')]:
            values = [r[source] for r in group]
            item[short + '_mean'] = statistics.mean(values)
            item[short + '_sd'] = statistics.stdev(values)
        item['icl_gain_vs_itransformer_pct'] = 100 * (1 - item['icl_only_mean'] / item['itransformer_base_mean'])
        item['combo_gain_vs_tefl_pct'] = 100 * (1 - item['tefl_icl_mean'] / item['tefl_only_mean'])
        summary.append(item)
    return summary


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 6.25)
    ax.axis('off')
    edges = [0.02, 0.28, 0.52, 0.76, 0.98]
    header_y = 5.65
    for j, text in enumerate(('Dataset / error cases', *METHODS)):
        left, right = edges[j], edges[j + 1]
        ax.add_patch(Rectangle((left, header_y), right-left-0.01, .42,
                               facecolor='#e9edf2', edgecolor='none'))
        ax.text(left + .016, header_y + .21, text, va='center', ha='left',
                fontsize=11, weight='bold', color='#1b2938')
    colors = {'ICL only': '#2364a6', 'TEFL only': '#596775', 'TEFL + ICL': '#b55c25'}
    for i, row in enumerate(summary):
        y = 4.62 - i * .91
        if i % 2 == 0:
            ax.add_patch(Rectangle((edges[0], y-.31), edges[-1]-edges[0], .77,
                                   facecolor='#f8f9fb', edgecolor='none'))
        ax.text(edges[0]+.016, y+.16, row['dataset'], ha='left', va='center',
                fontsize=11, weight='bold', color='#17212b')
        ax.text(edges[0]+.016, y-.11, row['history_rule'], ha='left', va='center',
                fontsize=8.5, color='#5f6b76')
        means = [row['icl_only_mean'], row['tefl_only_mean'], row['tefl_icl_mean']]
        sds = [row['icl_only_sd'], row['tefl_only_sd'], row['tefl_icl_sd']]
        best = min(range(3), key=lambda k: means[k])
        gains = [row['icl_gain_vs_itransformer_pct'], None,
                 row['combo_gain_vs_tefl_pct']]
        for j, method in enumerate(METHODS):
            left, right = edges[j+1], edges[j+2]
            if j == best:
                ax.add_patch(Rectangle((left+.005, y-.29), right-left-.02, .73,
                                       facecolor='#e5f4ec', edgecolor='none'))
            ax.text(left+.016, y+.15, f'{means[j]:.4f} ± {sds[j]:.4f}',
                    ha='left', va='center', color=colors[method], fontsize=11,
                    weight='bold' if j == best else 'normal')
            if gains[j] is not None:
                base = 'iTransformer' if j == 0 else 'TEFL'
                ax.text(left+.016, y-.13, f'{gains[j]:+.2f}% vs {base}',
                        ha='left', va='center', color='#57636f', fontsize=8.6)
    ax.plot([edges[0], edges[-1]], [.12, .12], color='#a7b0ba', lw=.7)
    ax.text(edges[0], -.02,
            'Test MSE (lower is better); mean ± SD, 3 seeds. L=96, H=L_error=192.',
            fontsize=9, color='#3c4854', ha='left', va='top')
    ax.text(edges[0], -.25,
            'ICL only: standalone iTransformer, zero-output head, case-wise W1+W2 with 10% W1 cap.',
            fontsize=8.5, color='#5f6b76', ha='left', va='top')
    ax.text(edges[0], -.45,
            'TEFL only and TEFL + ICL share a separately trained TEFL backbone; gains use each method\'s own baseline.',
            fontsize=8.5, color='#5f6b76', ha='left', va='top')
    fig.savefig(OUT / 'icl_tefl_three_way.png', dpi=300, bbox_inches='tight', pad_inches=.08)
    fig.savefig(OUT / 'icl_tefl_three_way.pdf', bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    summary = summarize(rows)
    write_csv(OUT / 'per_seed.csv', rows)
    write_csv(OUT / 'mean_sd.csv', summary)
    plot(summary)
    print(OUT / 'icl_tefl_three_way.png')


if __name__ == '__main__':
    main()
