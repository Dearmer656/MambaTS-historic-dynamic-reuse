"""Plots for err_corr_lag.py: (1) same-lead ACF vs target gap, (2) same-origin and (3) same-target corr vs lead gap; train/val/test."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = Path('patchmem/logs/server/err_corr')
seed = sys.argv[1] if len(sys.argv) > 1 else '3047'
tags = sys.argv[2].split(',') if len(sys.argv) > 2 else ['']                     # e.g. ',_unc,_raw'
SPL = ('train', 'val', 'test')
z = {(t, s): np.load(D / f'T_H192_L720_s{seed}{t}_{s}.npz') for t in tags for s in SPL}
LS = {'': '-', '_unc': '--', '_raw': ':'}
NAME = {'': 'z-score', '_unc': 'uncentered', '_raw': 'raw pooled'}
gap = lambda M: np.array([np.mean(np.diagonal(M, k)) for k in range(M.shape[0])])
fig, ax = plt.subplots(2, 3, figsize=(17, 8))
for j, h in enumerate((0, 23, 191)):
    for (t, s), v in z.items():
        ax[0, j].plot(v['A'][h], LS[t], lw=1, color=f'C{SPL.index(s)}', label=f'{s} {NAME[t]}')
    ax[0, j].set(title=f'same lead h={h}: corr(e_o[h], e_(o+d)[h])', xlabel='target gap d (h)')
for j, (key, ttl) in enumerate((('B', 'same origin: corr(e_o[i], e_o[j])'), ('Cm', 'same target: corr(e_(tau-i)[i], e_(tau-j)[j])'))):
    for (t, s), v in z.items():
        ax[1, j].plot(gap(v[key]), LS[t], lw=1, color=f'C{SPL.index(s)}', label=f'{s} {NAME[t]}')
    ax[1, j].set(title=ttl + ' vs |i-j|', xlabel='lead gap |i-j|')
for (t, s), v in z.items():
    ax[1, 2].plot(v['A'][191][:49], LS[t], marker='o', ms=2, lw=1, color=f'C{SPL.index(s)}', label=f'{s} {NAME[t]}')
ax[1, 2].set(title='zoom: same lead h=191, d <= 48', xlabel='target gap d (h)')
for a in ax.flat:
    a.axhline(0, color='k', lw=.5)
    for v in (24, 48, 72, 96, 120, 144, 168, 336, 504):
        if v <= a.get_xlim()[1]:
            a.axvline(v, color='grey', lw=.4, ls=':')
    a.legend(fontsize=7)
fig.tight_layout()
out = D / f'err_corr_lag_s{seed}{"".join(tags) if tags != [""] else ""}.png'
fig.savefig(out, dpi=90)
print(out)
