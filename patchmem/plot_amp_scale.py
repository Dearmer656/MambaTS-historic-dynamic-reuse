"""Val MSE vs alpha (diag_amp_scale.py jsons): rows = datasets, cols = raw / centered, one line per seed."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path('patchmem/logs/server/amp_scale')
NAMES = {'W': 'Weather', 'T': 'Traffic', 'E': 'ETTh2', 'M1': 'ETTm1'}
SEEDS = (3047, 2024, 2025)


def main(H=192):
    fig, ax = plt.subplots(len(NAMES), 2, figsize=(11, 3.2 * len(NAMES)), squeeze=False)
    rows = []
    for i, (c, name) in enumerate(NAMES.items()):
        for j, k in enumerate(('raw', 'centered')):
            a_ = ax[i, j]
            for s in SEEDS:
                p = D / f'{c}_H{H}_s{s}.json'
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                a_.plot(r['alphas'], r[k]['val_curve'], label=f's{s}: alpha {r[k]["alpha"]:.3f}, test {r[k]["gain"]:+.2f}%')
                a_.axvline(r[k]['alpha'], ls=':', lw=0.8, color=a_.lines[-1].get_color())
                rows.append((name, k, s, r[k]['alpha'], r[k]['gain']))
            a_.axvline(1.0, color='k', lw=0.6)
            a_.set_title(f'{name}: {k} scaling (val MSE)')
            a_.set_xlabel('alpha')
            a_.legend(fontsize=8)
    out = D / 'figures'
    out.mkdir(exist_ok=True)
    p = out / f'amp_scale_val_H{H}.png'
    fig.savefig(p, dpi=100, bbox_inches='tight')
    print(p)
    for r in rows:
        print('%-8s %-8s s%d alpha %.3f test gain %+.3f%%' % r)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 192)
