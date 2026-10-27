"""Plot stride/history performance from official-test iTransformer-TEFL ICL JSONs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='patchmem/logs/server/itr_tefl_icl')
    parser.add_argument('--output_dir', default='patchmem/logs/server/itr_tefl_icl/figures')
    parser.add_argument('--seed', type=int, default=3047)
    args = parser.parse_args()
    src, dst = Path(args.input_dir), Path(args.output_dir)
    data = {code: json.loads((src / f'{code}_s{args.seed}.json').read_text())
            for code in ('E', 'W', 'T') if (src / f'{code}_s{args.seed}.json').exists()}
    if not data:
        raise FileNotFoundError(f'No completed result JSONs in {src}')
    codes = list(data)
    dst.mkdir(parents=True, exist_ok=True)

    rows = []
    for code, result in data.items():
        for cell in result['test'].values():
            for family, item in cell['arms'].items():
                rows.append(dict(dataset=result['name'], seed=result['seed'], L=result['L'], H=result['H'],
                                 N=cell['N'], stride=cell['stride'], n_demos=cell['n_demos'],
                                 family=family, a1=item['a1'], a2=item['a2'],
                                 tefl_mse=cell['tefl_mse'], updated_mse=item['mse'], gain_pct=item['gain_pct']))
    with (dst / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({'font.size': 10, 'font.family': 'serif',
                         'font.serif': ['DejaVu Serif'], 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 300})
    fig, axes = plt.subplots(1, len(codes), figsize=(3.5 * len(codes), 3.2), sharey=False)
    if len(codes) == 1:
        axes = [axes]
    palette = {1440: '#0072B2', 2880: '#D55E00'}
    for ax, code in zip(axes, codes):
        result = data[code]
        Ns = sorted({cell['N'] for cell in result['test'].values()})
        for N in Ns:
            cells = sorted((cell for cell in result['test'].values() if cell['N'] == N),
                           key=lambda cell: cell['stride'])
            ax.plot([cell['stride'] for cell in cells],
                    [cell['arms']['W1+W2']['gain_pct'] for cell in cells],
                    marker='o', markersize=4, linewidth=1.8,
                    color=palette.get(N), label=f'History {N}')
        ax.axhline(0, color='0.45', linewidth=0.8, linestyle='--')
        ax.set_title(result['name'])
        ax.set_xlabel('Demo stride (time steps)')
        ax.set_xticks(sorted({cell['stride'] for cell in result['test'].values()}))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Test MSE improvement over TEFL (%)')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(dst / f'gain_by_stride_history.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Absolute MSE is provided separately so percentage gains cannot hide a
    # weak base forecast or a scale difference across datasets.
    fig, axes = plt.subplots(1, len(codes), figsize=(3.5 * len(codes), 3.2))
    if len(codes) == 1:
        axes = [axes]
    for ax, code in zip(axes, codes):
        result = data[code]
        cells = result['test'].values()
        base = next(iter(cells))['tefl_mse']
        ax.axhline(base, color='0.35', linewidth=1.4, linestyle='--', label='TEFL')
        for N in sorted({cell['N'] for cell in cells}):
            picked = sorted((cell for cell in result['test'].values() if cell['N'] == N),
                            key=lambda cell: cell['stride'])
            ax.plot([cell['stride'] for cell in picked],
                    [cell['arms']['W1+W2']['mse'] for cell in picked],
                    marker='o', markersize=4, linewidth=1.8,
                    color=palette.get(N), label=f'History {N}')
        ax.set_title(result['name'])
        ax.set_xlabel('Demo stride (time steps)')
        ax.set_xticks(sorted({cell['stride'] for cell in result['test'].values()}))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Official test MSE')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(dst / f'mse_by_stride_history.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'Wrote {dst}/gain_by_stride_history.pdf, mse_by_stride_history.pdf, results.csv')


if __name__ == '__main__':
    main()
