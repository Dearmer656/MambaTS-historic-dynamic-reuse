"""Plots for err_corr_lag.py: (1) same-lead ACF vs target gap, (2) same-origin and (3) same-target corr vs lead gap; train/val/test."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = Path('patchmem/logs/server/err_corr')
seed = sys.argv[1] if len(sys.argv) > 1 else '3047'
SPL = ('train', 'val', 'test')
z = {s: np.load(D / f'T_H192_L720_s{seed}_{s}.npz') for s in SPL}
gap = lambda M: np.array([np.mean(np.diagonal(M, k)) for k in range(M.shape[0])])
fig, ax = plt.subplots(2, 3, figsize=(17, 8))
for j, h in enumerate((0, 23, 191)):
    for s in SPL:
        ax[0, j].plot(z[s]['A'][h], lw=1, label=s)
    ax[0, j].set(title=f'same lead h={h}: corr(e_o[h], e_(o+d)[h])', xlabel='target gap d (h)')
for j, (key, t) in enumerate((('B', 'same origin: corr(e_o[i], e_o[j])'), ('Cm', 'same target: corr(e_(tau-i)[i], e_(tau-j)[j])'))):
    for s in SPL:
        ax[1, j].plot(gap(z[s][key]), lw=1, label=s)
    ax[1, j].set(title=t + ' vs |i-j|', xlabel='lead gap |i-j|')
for s in SPL:
    ax[1, 2].plot(z[s]['A'][191][:49], 'o-', ms=2, lw=1, label=f'{s} h191')
ax[1, 2].set(title='zoom: same lead h=191, d <= 48', xlabel='target gap d (h)')
for a in ax.flat:
    a.axhline(0, color='k', lw=.5)
    for v in (24, 48, 72, 96, 120, 144, 168, 336, 504):
        if v <= a.get_xlim()[1]:
            a.axvline(v, color='grey', lw=.4, ls=':')
    a.legend(fontsize=7)
fig.tight_layout()
out = D / f'err_corr_lag_s{seed}.png'
fig.savefig(out, dpi=90)
print(out)
