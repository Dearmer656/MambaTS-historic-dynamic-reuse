"""Plot the linear mimic A of the iTransformer-TEFL map (diag_tefl_map.py npz): heatmap and top-4 SVD directions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = Path('patchmem/logs/server/tefl_map')
NAMES = {'E': 'ETTh2', 'M1': 'ETTm1', 'M2': 'ETTm2', 'W': 'Weather', 'T': 'Traffic'}
STEP = {'E': '1 h', 'M1': '15 min', 'M2': '15 min', 'W': '10 min', 'T': '1 h'}


def main(seed=3047):
    fig, ax = plt.subplots(5, 3, figsize=(15, 20))
    for i, (c, name) in enumerate(NAMES.items()):
        A = np.load(D / f'{c}_tefl_map_s{seed}.npz')['A']                  # in j x out h, g = eps @ A
        js = json.loads((D / f'{c}_tefl_map_s{seed}.json').read_text())
        a = js['arms']
        v = np.abs(A).max()
        im = ax[i, 0].imshow(A, cmap='RdBu_r', vmin=-v, vmax=v, origin='lower', aspect='auto')
        ax[i, 0].set_xlabel('output lead h (target t+h)')
        ax[i, 0].set_ylabel('input position j of eps (time t-192+j)')
        ax[i, 0].set_title(f'{name} ({STEP[c]}): A, TEFL gain {js["tefl_gain_pct"]:.1f}%, lin keeps '
                           f'{100 * a["lin_mimic"]["retained"]:.0f}%')
        fig.colorbar(im, ax=ax[i, 0], fraction=0.046)
        U, S, Vh = np.linalg.svd(A)
        share = S ** 2 / (S ** 2).sum()
        for k in range(4):
            sg = np.sign(U[:, k].sum()) or 1.0
            ax[i, 1].plot(U[:, k] * sg, lw=1.2, label=f'k={k + 1} ({100 * share[k]:.0f}%)')
            ax[i, 2].plot(Vh[k] * sg * S[k], lw=1.2, label=f'k={k + 1}')
        ax[i, 1].set_title('input direction (which part of eps is read)')
        ax[i, 1].set_xlabel('j')
        ax[i, 2].set_title('output profile x singular value (what is added)')
        ax[i, 2].set_xlabel('h')
        for j in (1, 2):
            ax[i, j].axhline(0, color='k', lw=0.5)
            ax[i, j].legend(fontsize=8)
    fig.tight_layout()
    out = D / 'figures'
    out.mkdir(exist_ok=True)
    fig.savefig(out / f'tefl_map_s{seed}.png', dpi=110)
    print(out / f'tefl_map_s{seed}.png')


if __name__ == '__main__':
    main()
