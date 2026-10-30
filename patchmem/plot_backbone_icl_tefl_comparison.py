"""PPT-ready four-way comparison using the existing three-seed test results."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


OUT = Path('patchmem/logs/server/icl_setting_sweep')
ORDER = ('ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Traffic')


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    picked = read(OUT / 'observed_best_per_seed.csv')
    grouped = defaultdict(list)
    for row in picked:
        grouped[row['dataset'], row['method']].append(row)
    summary = []
    for dataset in ORDER:
        pure = grouped[dataset, 'ICL only']
        joint = grouped[dataset, 'TEFL + ICL']
        assert len(pure) == len(joint) == 3
        values = {
            'iTransformer': [float(r['test_base_mse']) for r in pure],
            'ICL only': [float(r['test_mse']) for r in pure],
            'TEFL': [float(r['tefl_baseline_mse']) for r in joint],
            'TEFL + ICL': [float(r['test_mse']) for r in joint],
        }
        row = {'dataset': dataset}
        for name, numbers in values.items():
            key = name.lower().replace(' ', '_').replace('+', 'plus')
            row[key + '_mean'] = statistics.mean(numbers)
            row[key + '_sd'] = statistics.stdev(numbers)
        row['icl_gain_vs_backbone_pct'] = 100 * (1 - row['icl_only_mean'] / row['itransformer_mean'])
        row['icl_gain_vs_tefl_pct'] = 100 * (1 - row['icl_only_mean'] / row['tefl_mean'])
        row['joint_gain_vs_tefl_pct'] = 100 * (1 - row['tefl_plus_icl_mean'] / row['tefl_mean'])
        row['icl_cell'] = pure[0]['cell']
        row['joint_cell'] = joint[0]['cell']
        summary.append(row)
    with (OUT / 'backbone_icl_tefl_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-.35, 6.35)
    ax.axis('off')
    edges = [.02, .19, .39, .59, .79, .99]
    heads = ['Dataset', 'iTransformer', 'ICL only', 'TEFL', 'TEFL + ICL']
    for j, title in enumerate(heads):
        left, right = edges[j:j + 2]
        ax.add_patch(Rectangle((left, 5.68), right - left - .007, .47,
                               facecolor='#e7ecf2', edgecolor='none'))
        ax.text(left + .012, 5.915, title, va='center', weight='bold', fontsize=12)
    keys = ['itransformer', 'icl_only', 'tefl', 'tefl_plus_icl']
    for i, row in enumerate(summary):
        y = 4.78 - .9 * i
        if i % 2 == 0:
            ax.add_patch(Rectangle((edges[0], y - .34), edges[-1] - edges[0], .78,
                                   facecolor='#f7f9fb', edgecolor='none'))
        ax.text(edges[0] + .012, y + .03, row['dataset'], va='center', weight='bold', fontsize=12)
        best = min(keys, key=lambda k: row[k + '_mean'])
        for j, key in enumerate(keys):
            left, right = edges[j + 1:j + 3]
            if key == best:
                ax.add_patch(Rectangle((left + .004, y - .31), right - left - .014, .72,
                                       facecolor='#dff2e6', edgecolor='none'))
            ax.text(left + .012, y + .10,
                    f"{row[key + '_mean']:.4f} ± {row[key + '_sd']:.4f}",
                    va='center', fontsize=10.8, weight='bold' if key == best else 'normal')
            if key == 'icl_only':
                detail = f"{row['icl_gain_vs_backbone_pct']:+.2f}% vs backbone"
            elif key == 'tefl_plus_icl':
                detail = f"{row['joint_gain_vs_tefl_pct']:+.2f}% vs TEFL"
            else:
                detail = ''
            if detail:
                ax.text(left + .012, y - .13, detail, va='center', fontsize=8.9,
                        color='#516374')
    fig.text(.04, .055,
             'ICL only improves on iTransformer in 5/5 datasets; it trails TEFL in ETTm1, ETTm2, and Weather.',
             fontsize=10.2, weight='bold', color='#22384b')
    fig.text(.04, .015,
             'Mean ± SD over 3 seeds. Best observed ICL setting per dataset selected by test MSE; exploratory comparison.',
             fontsize=9, color='#596673')
    fig.savefig(OUT / 'backbone_icl_tefl_comparison.png', dpi=300,
                bbox_inches='tight', pad_inches=.08)
    fig.savefig(OUT / 'backbone_icl_tefl_comparison.pdf',
                bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    for row in summary:
        print(row['dataset'], *(f"{row[k + '_mean']:.4f}" for k in keys))


if __name__ == '__main__':
    main()
