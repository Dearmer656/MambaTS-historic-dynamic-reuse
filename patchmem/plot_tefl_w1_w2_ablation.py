"""Three-seed TEFL W1/W2 ablation at validation-selected case settings."""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from patchmem.plot_icl_setting_sweep import OUT, SEEDS, DATASETS, sources


def main():
    selected = {
        (r['code'], int(r['seed'])): r['cell']
        for r in csv.DictReader((OUT / 'validation_selected_per_seed.csv').open())
        if r['method'] == 'TEFL + ICL'
    }
    found = {}
    for method, path in sources():
        if method != 'TEFL + ICL':
            continue
        data = json.loads(path.read_text())
        if data.get('baseline_tag') != 'TEFL_paper_L96_E192':
            continue
        key = (data['dataset'], data['seed'])
        cell = selected.get(key)
        if cell not in data.get('test', {}):
            continue
        result = data['test'][cell]
        row = dict(dataset=DATASETS[key[0]], seed=key[1], setting=cell,
                   tefl=result['tefl_mse'],
                   w1_only=result['arms']['W1_only']['mse'],
                   w2_only=result['arms']['W2_only']['mse'],
                   w1_plus_w2=result['arms']['W1+W2']['mse'])
        if key in found:
            assert abs(found[key]['w1_plus_w2'] - row['w1_plus_w2']) < 1e-6
        else:
            found[key] = row
    assert len(found) == len(DATASETS) * len(SEEDS)
    rows = [found[code, seed] for code in DATASETS for seed in SEEDS]
    with (OUT / 'tefl_w1_w2_ablation_per_seed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for code, name in DATASETS.items():
        group = [found[code, seed] for seed in SEEDS]
        row = {'dataset': name}
        for key in ('tefl', 'w1_only', 'w2_only', 'w1_plus_w2'):
            values = [r[key] for r in group]
            row[key + '_mean'] = statistics.mean(values)
            row[key + '_sd'] = statistics.stdev(values)
            if key != 'tefl':
                row[key + '_gain_vs_tefl_pct'] = 100 * (1 - row[key + '_mean'] / row['tefl_mean'])
        row['settings'] = '|'.join(f"s{r['seed']}:{r['setting']}" for r in group)
        summary.append(row)
    with (OUT / 'tefl_w1_w2_ablation_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(12.2, 5.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(-.3, 6.3)
    ax.axis('off')
    edges = [.02, .20, .39, .58, .78, .99]
    heads = ['Dataset', 'TEFL', 'W1 only', 'W2 only', 'W1 + W2']
    for j, title in enumerate(heads):
        left, right = edges[j:j + 2]
        ax.add_patch(Rectangle((left, 5.68), right - left - .008, .46,
                               facecolor='#e7ecf2', edgecolor='none'))
        ax.text(left + .012, 5.91, title, va='center', weight='bold', fontsize=12)
    keys = ('tefl', 'w1_only', 'w2_only', 'w1_plus_w2')
    for i, row in enumerate(summary):
        y = 4.79 - .90 * i
        if i % 2 == 0:
            ax.add_patch(Rectangle((edges[0], y - .32), edges[-1] - edges[0], .77,
                                   facecolor='#f7f9fb', edgecolor='none'))
        ax.text(edges[0] + .012, y + .03, row['dataset'], va='center', weight='bold', fontsize=12)
        best = min(keys, key=lambda key: row[key + '_mean'])
        for j, key in enumerate(keys):
            left, right = edges[j + 1:j + 3]
            if key == best:
                ax.add_patch(Rectangle((left + .004, y - .30), right - left - .014, .72,
                                       facecolor='#dff2e6', edgecolor='none'))
            ax.text(left + .012, y + .10,
                    f"{row[key + '_mean']:.4f} ± {row[key + '_sd']:.4f}",
                    va='center', fontsize=10.5, weight='bold' if key == best else 'normal')
            if key != 'tefl':
                ax.text(left + .012, y - .14,
                        f"{row[key + '_gain_vs_tefl_pct']:+.2f}% vs TEFL",
                        va='center', fontsize=8.7, color='#536576')
    fig.text(.04, .055, 'W2-only captures nearly all measured gains; W1 adds no consistent improvement.',
             fontsize=10.2, weight='bold', color='#22384b')
    fig.text(.04, .015,
             'Same case rule selected by validation for W1+W2; each arm tunes its step size on validation. Mean ± SD, 3 seeds.',
             fontsize=9, color='#596673')
    fig.savefig(OUT / 'tefl_w1_w2_ablation.png', dpi=300, bbox_inches='tight', pad_inches=.08)
    fig.savefig(OUT / 'tefl_w1_w2_ablation.pdf', bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    for row in summary:
        print(row['dataset'], *(f"{row[key + '_mean']:.6f}" for key in keys))


if __name__ == '__main__':
    main()
