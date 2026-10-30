"""Five setting-sweep figures and an exploratory best-vs-TEFL table.

Every plotted point is a three-seed test result. The step size within each
setting was selected on validation by the original evaluator. When multiple
fixed strides exist at one history length, validation chooses the stride per
seed. The final table chooses one common setting per method and dataset by
three-seed mean test MSE; validation-selected alternatives are exported too.
"""
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path('patchmem/logs/server')
OUT = ROOT / 'icl_setting_sweep'
SEEDS = (2024, 2025, 3047)
DATASETS = {'E': 'ETTh2', 'M1': 'ETTm1', 'M2': 'ETTm2',
            'W': 'Weather', 'T': 'Traffic'}
PERIODS = {'E': (24, 168), 'M1': (96, 672), 'M2': (96, 672),
           'W': (144, 1008), 'T': (24, 168)}
CATEGORIES = ('Full lookback', 'Periodic day', 'Periodic week', 'Fixed stride')
COLORS = {'Full lookback': '#444444', 'Periodic day': '#277da1',
          'Periodic week': '#de8f05', 'Fixed stride': '#8b5fbf'}


def category(code, key):
    match = re.fullmatch(r'N(\d+)_([DP])(\d+)', key)
    if not match:
        return None
    n, mode, step = int(match[1]), match[2], int(match[3])
    if mode == 'D':
        return n, step, 'Full lookback' if step == 1 else 'Fixed stride'
    day, week = PERIODS[code]
    if step == day:
        return n, step, 'Periodic day'
    if step == week:
        return n, step, 'Periodic week'
    return None


def sources():
    tefl_dir = ROOT / 'itr_tefl_icl'
    for path in tefl_dir.glob('*.json'):
        if re.search(r'_s(?:2024|2025|3047)\.json$', path.name):
            yield 'TEFL + ICL', path
    pure_dir = ROOT / 'itr_pure_icl'
    for path in pure_dir.glob('*_pure_icl_clip0p1_s*.json'):
        yield 'ICL only', path
    supp_dir = ROOT / 'itr_icl_supp'
    for path in supp_dir.glob('*.json'):
        method = 'ICL only' if '_pure_' in path.name else 'TEFL + ICL'
        yield method, path


def load_rows():
    found = {}
    for method, path in sources():
        data = json.loads(path.read_text())
        if method == 'TEFL + ICL' and data.get('baseline_tag') != 'TEFL_paper_L96_E192':
            continue
        if method == 'ICL only' and data.get('clip_w1') != .1:
            continue
        code, seed = data['dataset'], data['seed']
        if code not in DATASETS or seed not in SEEDS:
            continue
        assert (data['L'], data['H'], data['L_error']) == (96, 192, 192)
        for key, result in data['test'].items():
            info = category(code, key)
            if info is None:
                continue
            n, step, kind = info
            val = data['validation'][key]
            field = 'base_mse' if method == 'ICL only' else 'tefl_mse'
            row = dict(dataset=DATASETS[code], code=code, method=method,
                       category=kind, history=n, step=step, cell=key,
                       seed=seed, n_demos=result['n_demos'],
                       test_windows=data['test_windows'],
                       val_base_mse=val[field],
                       val_mse=val['chosen']['W1+W2']['mse'],
                       test_base_mse=result[field],
                       test_mse=result['arms']['W1+W2']['mse'])
            id_ = (code, method, key, seed)
            if id_ in found:
                old = found[id_]
                if abs(old['test_mse'] - row['test_mse']) > 1e-6:
                    raise AssertionError(f'Duplicate setting differs: {path} {id_}')
            else:
                found[id_] = row
    return list(found.values())


def baseline_rows():
    baselines = {}
    for code in DATASETS:
        for seed in SEEDS:
            path = ROOT / 'itr_tefl_icl' / f'{code}_TEFL_paper_L96_E192_s{seed}.json'
            data = json.loads(path.read_text())
            baseline = next(iter(data['test'].values()))['tefl_mse']
            baselines[code, seed] = baseline
    return baselines


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped_points(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['code'], row['method'], row['category'], row['history'], row['seed']].append(row)
    selected = []
    for key, candidates in buckets.items():
        chosen = min(candidates, key=lambda r: r['val_mse'])
        selected.append(chosen)
    groups = defaultdict(list)
    for row in selected:
        groups[row['code'], row['method'], row['category'], row['history']].append(row)
    return groups, selected


def draw_dataset(code, groups, tefl_baselines):
    name = DATASETS[code]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharey=False)
    tefl_mean = statistics.mean(tefl_baselines[code, s] for s in SEEDS)
    pure_rows = [r for (c, m, _, _), records in groups.items()
                 if c == code and m == 'ICL only' for r in records]
    itr_mean = statistics.mean(next(r['test_base_mse'] for r in pure_rows if r['seed'] == s)
                               for s in SEEDS)
    for ax, method in zip(axes, ('TEFL + ICL', 'ICL only')):
        for kind in CATEGORIES:
            points = []
            for (c, m, k, n), rows in groups.items():
                if c != code or m != method or k != kind or len(rows) != len(SEEDS):
                    continue
                mse = [r['test_mse'] for r in rows]
                points.append((n, statistics.mean(mse), statistics.stdev(mse)))
            if not points:
                continue
            points.sort()
            xs, means, sds = zip(*points)
            ax.errorbar(xs, means, yerr=sds, marker='o', ms=5,
                        linestyle='-' if len(xs) > 1 else 'none', lw=1.5,
                        capsize=2.5, color=COLORS[kind], label=kind)
        ax.axhline(tefl_mean, color='#334e68', lw=1, ls='--', label='TEFL baseline')
        if method == 'ICL only':
            ax.axhline(itr_mean, color='#7b8791', lw=1, ls=':', label='iTransformer baseline')
        ax.set_title(method, fontsize=11)
        ax.set_xscale('log', base=2)
        ax.grid(axis='y', color='#d9dfe5', linewidth=.6)
        ax.set_xlabel('Error-case history length N')
        xs = sorted({n for (c,m,_,n) in groups if c == code and m == method})
        if xs:
            ax.set_xticks(xs, [str(n) for n in xs], rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Test MSE (lower is better)')
    axes[1].set_ylabel('Test MSE (lower is better)')
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.91),
               ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f'{name} — L=96, H=L_error=192; 3 seeds', fontsize=12, y=.99)
    note = ('Full lookback uses every eligible error case (stride 1); '
            'fixed-stride point uses validation-selected stride.')
    if code == 'T':
        note += ' ICL-only full/day incomplete; omitted.'
    fig.text(.5, .025, note,
             ha='center', fontsize=8, color='#4f5b66')
    fig.tight_layout(rect=(0, .09, 1, .85))
    fig.savefig(OUT / f'{name.lower()}_settings.png', dpi=250)
    fig.savefig(OUT / f'{name.lower()}_settings.pdf')
    plt.close(fig)


def best_table(rows, baselines):
    picked = []
    validation_picked = []
    for code in DATASETS:
        for method in ('TEFL + ICL', 'ICL only'):
            by_cell = defaultdict(list)
            for row in rows:
                if row['code'] == code and row['method'] == method:
                    by_cell[row['cell']].append(row)
            complete = {cell: group for cell, group in by_cell.items()
                        if {r['seed'] for r in group} == set(SEEDS)}
            if complete:
                best_cell = min(complete, key=lambda cell: statistics.mean(
                    r['test_mse'] for r in complete[cell]))
                for chosen in complete[best_cell]:
                    seed = chosen['seed']
                    picked.append({**chosen, 'tefl_baseline_mse': baselines[code, seed],
                                   'gain_vs_tefl_pct': 100 * (1 - chosen['test_mse']/baselines[code,seed])})
            for seed in SEEDS:
                candidates = [r for group in by_cell.values() for r in group if r['seed'] == seed]
                if candidates:
                    chosen = min(candidates, key=lambda r: r['val_mse'])
                    validation_picked.append({**chosen, 'tefl_baseline_mse': baselines[code, seed],
                                              'gain_vs_tefl_pct': 100 * (1 - chosen['test_mse']/baselines[code,seed])})
    write_csv(OUT / 'observed_best_per_seed.csv', picked)
    write_csv(OUT / 'validation_selected_per_seed.csv', validation_picked)
    validation_summary = []
    for code, name in DATASETS.items():
        tefl = [baselines[code, s] for s in SEEDS]
        result = dict(dataset=name, tefl_mean=statistics.mean(tefl),
                      tefl_sd=statistics.stdev(tefl))
        for method, prefix in [('TEFL + ICL', 'tefl_icl'), ('ICL only', 'icl_only')]:
            group = [r for r in validation_picked
                     if r['code'] == code and r['method'] == method]
            test = [r['test_mse'] for r in group]
            result[prefix + '_mean'] = statistics.mean(test)
            result[prefix + '_sd'] = statistics.stdev(test)
            result[prefix + '_gain_vs_tefl_pct'] = 100 * (1 - result[prefix + '_mean'] / result['tefl_mean'])
            result[prefix + '_selected_cells'] = '|'.join(
                f"s{r['seed']}:{r['cell']}" for r in sorted(group, key=lambda x: x['seed']))
        validation_summary.append(result)
    write_csv(OUT / 'validation_selected_vs_tefl.csv', validation_summary)
    summary = []
    for code, name in DATASETS.items():
        tefl = [baselines[code,s] for s in SEEDS]
        result = dict(dataset=name, tefl_mean=statistics.mean(tefl),
                      tefl_sd=statistics.stdev(tefl))
        for method, prefix in [('TEFL + ICL','tefl_icl'),('ICL only','icl_only')]:
            group = [r for r in picked if r['code']==code and r['method']==method]
            if len(group) == len(SEEDS):
                test = [r['test_mse'] for r in group]
                result[prefix+'_mean'] = statistics.mean(test)
                result[prefix+'_sd'] = statistics.stdev(test)
                result[prefix+'_gain_vs_tefl_pct'] = 100 * (1 - result[prefix+'_mean']/result['tefl_mean'])
                result[prefix+'_selected_cells'] = '|'.join(f"s{r['seed']}:{r['cell']}" for r in sorted(group,key=lambda x:x['seed']))
        summary.append(result)
    write_csv(OUT / 'best_observed_vs_tefl.csv', summary)
    draw_table(summary)
    write_report(summary, validation_summary)
    return summary


def write_report(observed, validation):
    lines = [
        '# iTransformer TEFL and case-wise ICL setting sweep',
        '',
        'Official setup: L=96, H=L_error=192; seeds 2024, 2025, 3047. '
        'All values below are mean test MSE over three seeds.',
        '',
        '| Dataset | TEFL | TEFL+ICL, observed best | ICL only, observed best | '
        'TEFL+ICL, validation selected | ICL only, validation selected |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    by_name = {r['dataset']: r for r in validation}
    for row in observed:
        val = by_name[row['dataset']]
        lines.append('| ' + ' | '.join([
            row['dataset'],
            f"{row['tefl_mean']:.6f}",
            f"{row['tefl_icl_mean']:.6f}",
            f"{row['icl_only_mean']:.6f}",
            f"{val['tefl_icl_mean']:.6f}",
            f"{val['icl_only_mean']:.6f}",
        ]) + ' |')
    lines += [
        '',
        'Observed best selects one common setting for each dataset and method by '
        'three-seed test mean. It is exploratory because test data were used for '
        'setting selection. The validation-selected columns choose a setting per '
        'seed using only validation MSE. Within every setting, W1/W2 step sizes '
        'were selected on validation.',
        '',
        'The ICL-only method uses a separately trained iTransformer, fixed random '
        'W1, zero W2, sequential W2 then W1 case updates, and a 10% W1 step cap. '
        'TEFL+ICL uses jointly trained TEFL checkpoints and TEFL prediction '
        'residuals in its case updates. Traffic TEFL+ICL full lookback was '
        'tested at N=576; ICL-only full/day runs were canceled before three '
        'seeds completed, so these points are omitted from figures and tables.',
        '',
        'The five `*_settings.png` figures show full lookback, periodic day/week, '
        'and fixed-stride curves where measured. `best_vs_tefl.png` shows the '
        'exploratory table. `all_settings_per_seed.csv` contains every evaluated '
        'setting and seed. `plot_points_per_seed.csv`, '
        '`observed_best_per_seed.csv`, and `validation_selected_per_seed.csv` '
        'record the selections behind the figures and summaries.',
    ]
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')


def draw_table(summary):
    fig, ax = plt.subplots(figsize=(11.4, 4.9))
    ax.set_xlim(0, 1); ax.set_ylim(.6, 6.2); ax.axis('off')
    edges = [.02, .22, .46, .72, .98]
    heads = ['Dataset', 'TEFL', 'Best TEFL + ICL', 'Best ICL only']
    for j, title in enumerate(heads):
        l,r=edges[j],edges[j+1]
        ax.add_patch(Rectangle((l,5.7),r-l-.01,.42,facecolor='#e8ecf0',edgecolor='none'))
        ax.text(l+.012,5.91,title,ha='left',va='center',weight='bold',fontsize=11)
    for i,row in enumerate(summary):
        y=4.8-i*.91
        if i%2==0:
            ax.add_patch(Rectangle((edges[0],y-.31),edges[-1]-edges[0],.76,
                                   facecolor='#f8f9fb',edgecolor='none'))
        ax.text(edges[0]+.012,y+.11,row['dataset'],ha='left',va='center',weight='bold',fontsize=11)
        means=[row['tefl_mean'],row.get('tefl_icl_mean'),row.get('icl_only_mean')]
        sds=[row['tefl_sd'],row.get('tefl_icl_sd'),row.get('icl_only_sd')]
        best=min((j for j,x in enumerate(means) if x is not None),key=lambda j:means[j])
        for j in range(3):
            if means[j] is None:continue
            l,r=edges[j+1],edges[j+2]
            if j==best:
                ax.add_patch(Rectangle((l+.004,y-.29),r-l-.014,.73,
                                       facecolor='#e3f2e9',edgecolor='none'))
            ax.text(l+.012,y+.15,f'{means[j]:.4f} ± {sds[j]:.4f}',
                    ha='left',va='center',fontsize=10.5,weight='bold' if j==best else 'normal')
            if j:
                prefix='tefl_icl' if j==1 else 'icl_only'
                gain=row.get(prefix+'_gain_vs_tefl_pct')
                ax.text(l+.012,y-.06,f'{gain:+.2f}% vs TEFL',ha='left',va='center',
                        fontsize=8.5,color='#596673')
                cells={part.split(':',1)[1] for part in row[prefix+'_selected_cells'].split('|')}
                label=next(iter(cells)) if len(cells)==1 else 'setting varies by seed'
                ax.text(l+.012,y-.24,label,ha='left',va='center',
                        fontsize=7.7,color='#6d7883')
    fig.text(.04,.02,'Best observed setting by 3-seed mean test MSE; step size within each setting was validation-selected. Exploratory comparison.',
             fontsize=9,color='#3c4854')
    fig.savefig(OUT/'best_vs_tefl.png',dpi=300,bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/'best_vs_tefl.pdf',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=load_rows()
    write_csv(OUT/'all_settings_per_seed.csv',rows)
    baselines=baseline_rows()
    groups,selected=grouped_points(rows)
    write_csv(OUT/'plot_points_per_seed.csv',selected)
    for code in DATASETS:
        draw_dataset(code,groups,baselines)
    best_table(rows,baselines)
    print(f'{len(rows)} setting-seed rows; output {OUT}')


if __name__=='__main__':
    main()
