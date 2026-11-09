"""Lead x lead error correlation (diag_err_lead.py npz): rows = datasets, cols = orig / diag / pred / resid."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = Path('patchmem/logs/server/err_lead')
NAMES = {'E': 'ETTh2', 'M1': 'ETTm1', 'M2': 'ETTm2', 'W': 'Weather', 'T': 'Traffic', 'EL': 'Electricity'}
KINDS = {'orig': 'one forecast E[o, h]', 'diag': 'same target E[t-1-l, l]', 'pred': 'predictable part',
         'resid': 'unpredictable part'}


def main(H=192, seed=3047):
    fig, ax = plt.subplots(len(NAMES), len(KINDS), figsize=(16, 24))
    for i, (c, name) in enumerate(NAMES.items()):
        z = np.load(D / f'{c}_H{H}_s{seed}.npz')
        st = json.loads((D / f'{c}_H{H}_s{seed}.json').read_text())['stats']
        for j, k in enumerate(KINDS):
            im = ax[i, j].imshow(z[k], cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
            s = st[k]
            ax[i, j].set_title(f'{name}: {KINDS[k]}\nR1 {s["R1"]:.2f}  R8 {s["R8"]:.2f}  PR {s["pr"]:.1f}', fontsize=10)
            ax[i, j].set_xlabel('lead' if k != 'diag' else 'l (forecast issued at t-1-l)')
    fig.colorbar(im, ax=ax, fraction=0.01)
    out = D / 'figures'
    out.mkdir(exist_ok=True)
    p = out / f'err_lead_corr_H{H}_s{seed}.png'
    fig.savefig(p, dpi=90, bbox_inches='tight')
    print(p)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 192)
