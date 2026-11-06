"""PAT-315: what does beta_t carry in P / Pfix?  From eval_sr --src dumps (val / test origin x channel cells).
Share of test beta variance explained by channel, hour-of-week, channel x hour-of-week (means fit on val), plus the distribution.
Usage: python -m patchmem.ana_beta --H 192
"""
import argparse
from pathlib import Path

import numpy as np

HOUR0_OFF = 4 * 24 + 2


def r2(y, pred):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--H', type=int, default=192)
    p.add_argument('--dir', default='patchmem/logs/server/err_src')
    p.add_argument('--seeds', default='3047,2024,2025')
    a = p.parse_args()
    for s in map(int, a.seeds.split(',')):
        for arm in ('s1_P', 's1_Pfix'):
            d = np.load(Path(a.dir) / f'T_H{a.H}_s{s}_{arm}_src.npz')
            sp = d['split']
            va, te = (sp == 1) | (sp == 3), sp == 2
            b = d['beta'].astype(np.float64)
            how = (d['times'] + HOUR0_OFF) % 168
            C = b.shape[1]
            bv, bt = b[va], b[te]
            ch_m = bv.mean(0)
            how_m = np.array([bv[how[va] == h].mean() for h in range(168)])
            chhow = np.stack([bv[how[va] == h].mean(0) for h in range(168)], 0)     # (168 C)
            ch_p = np.broadcast_to(ch_m, bt.shape)
            how_p = np.broadcast_to(how_m[how[te]][:, None], bt.shape)
            ac = np.corrcoef(bt[:-1].ravel(), bt[1:].ravel())[0, 1]
            q = np.quantile(bt, [.01, .1, .5, .9, .99])
            print(f'H{a.H} s{s} {arm:8s} test beta mean {bt.mean():.3f} std {bt.std():.3f} q01/10/50/90/99 '
                  + '/'.join(f'{x:.2f}' for x in q)
                  + f' | frac>1 {(bt > 1).mean():.3f} | R2 channel {r2(bt, ch_p):.3f} how {r2(bt, how_p):.3f} '
                  f'chxhow {r2(bt, chhow[how[te]]):.3f} | lag1 corr {ac:.3f} | C {C}')


if __name__ == '__main__':
    main()
