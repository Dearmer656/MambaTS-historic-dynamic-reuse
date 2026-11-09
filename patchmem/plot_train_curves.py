"""Carry train loss vs the zero-carry reference (carry_zero_ref.py) and early-stop val gain, Traffic L720 H192."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path('patchmem/logs/server/err_sr_mech')
ZERO = {3047: 0.19867, 2024: 0.19746, 2025: 0.19710}                              # mean E^2, train targets
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for s, z in ZERO.items():
    r = json.loads((D / f'T_H192_L720_s{s}.json').read_text())
    bv = r['base_sh']['val_mse']
    for arm, ls in (('s1_P', '-'), ('s1_GF', '--')):
        tl = r[arm]['train_loss']
        tr, es = tl[:-1], tl[-1]
        l0, = ax[0].plot(range(len(tr)), [100 * (z - x) / z for x in tr], ls, marker='o', ms=3, label=f'{arm} s{s}')
        ax[1].plot(range(len(es['val'])), [100 * (bv - v) / bv for v in es['val']], ls, marker='o', ms=3, color=l0.get_color())
        ax[1].plot(es['best_epoch'], 100 * (bv - es['val'][es['best_epoch']]) / bv, '*', ms=12, color=l0.get_color())
ax[0].axhline(0, color='k', lw=0.7)
ax[0].set(title='train loss reduction vs zero carry (%)', xlabel='epoch', ylabel='100 (E^2 - loss) / E^2')
ax[1].axhline(0, color='k', lw=0.7)
ax[1].set(title='val MSE gain vs base_sh per epoch (%), * = chosen', xlabel='epoch')
ax[0].legend(fontsize=7, ncol=2)
out = D / 'figures' / 'train_curves_H192.png'
fig.savefig(out, dpi=110, bbox_inches='tight')
print(out)
