"""Aggregate three-seed standalone iTransformer ICL ablations."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('patchmem/logs/server/itr_pure_icl')
OUT = ROOT / 'figures'
DATASETS = ['E', 'M1', 'M2', 'W', 'T']
LABELS = {'E': 'ETTh2', 'M1': 'ETTm1', 'M2': 'ETTm2', 'W': 'Weather', 'T': 'Traffic'}
BEST_CELLS = {'E': 'N1440_D48', 'M1': 'N2880_D96', 'M2': 'N2880_D96',
              'W': 'N5760_P144', 'T': 'N2880_P168'}


def main():
    rows = []
    for path in sorted(ROOT.glob('*_pure_icl*_s*.json')):
        result = json.loads(path.read_text())
        cap = result.get('clip_w1')
        for cell, entry in result['test'].items():
            for arm, metrics in entry['arms'].items():
                rows.append(dict(dataset=result['dataset'], seed=result['seed'],
                                 cell=cell, n_demos=entry['n_demos'],
                                 arm=arm, clip_w1='none' if cap is None else str(cap),
                                 a1=metrics['a1'], a2=metrics['a2'],
                                 base_mse=entry['base_mse'], mse=metrics['mse'],
                                 gain_pct=metrics['gain_pct'],
                                 val_prefers_update=metrics['val_prefers_update']))
    if not rows:
        raise RuntimeError(f'No result JSON in {ROOT}')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'seed_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['cell'], row['arm'], row['clip_w1'])].append(row)
    summary = []
    for (dataset, cell, arm, cap), group in sorted(groups.items()):
        gains = [x['gain_pct'] for x in group]
        summary.append(dict(dataset=dataset, cell=cell, arm=arm, clip_w1=cap,
                            seeds=len(group), mean_base_mse=statistics.mean(x['base_mse'] for x in group),
                            mean_mse=statistics.mean(x['mse'] for x in group),
                            mean_gain_pct=statistics.mean(gains),
                            sd_gain_pct=statistics.stdev(gains) if len(gains) > 1 else 0,
                            accepted_seeds=sum(x['val_prefers_update'] for x in group)))
    with (OUT / 'mean_sd.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    if not all(any(key[0] == dataset for key in groups) for dataset in DATASETS):
        print('Traffic or another dataset pending; CSVs written, chart deferred')
        return
    indexed = {(x['dataset'], x['cell'], x['arm'], x['clip_w1']): x for x in summary}
    fig, ax = plt.subplots(figsize=(10, 4.7))
    x = np.arange(len(DATASETS))
    configurations = [('W2 only', 'W2_only', 'none', '#90a4ae'),
                      ('W1+W2, free', 'W1+W2', 'none', '#f0a44b'),
                      ('W1+W2, W1 cap 10%', 'W1+W2', '0.1', '#457bba')]
    width = 0.24
    for i, (label, arm, cap, color) in enumerate(configurations):
        means, sds = [], []
        for dataset in DATASETS:
            value = indexed[(dataset, BEST_CELLS[dataset], arm, cap)]
            means.append(value['mean_gain_pct'])
            sds.append(value['sd_gain_pct'])
        xpos = x + (i - 1) * width
        visible = np.clip(means, -6, 8)
        ax.bar(xpos, visible, width, label=label, color=color)
        for j, mean in enumerate(means):
            if mean < -6:
                label = f'{mean/1000:.0f}k%' if abs(mean) >= 10000 else f'{mean:.0f}%'
                ax.annotate(label, (xpos[j], -5.8), ha='center', va='bottom',
                            rotation=90, fontsize=8, color='#8a3c00')
            else:
                ax.errorbar(xpos[j], mean, yerr=sds[j], fmt='none',
                            ecolor='#303030', capsize=2, linewidth=1)
    ax.axhline(0, color='#333333', linewidth=0.8)
    ax.set_xticks(x, [LABELS[d] for d in DATASETS])
    ax.set_ylabel('Test MSE improvement over standalone iTransformer (%)')
    ax.set_ylim(-6, 8)
    ax.set_title('No TEFL: case-wise ICL from zero-output two-layer head (3 seeds)')
    ax.legend(loc='upper left', ncol=3, fontsize=8)
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / 'gain_by_dataset.png', dpi=200)
    plt.close(fig)
    print(OUT / 'gain_by_dataset.png')


if __name__ == '__main__':
    main()
