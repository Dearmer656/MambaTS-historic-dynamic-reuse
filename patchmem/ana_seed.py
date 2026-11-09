"""PAT-315: is *where* the carry gains stable across carry seeds, although beta_t timing is not?
eval_sr --src dumps of Pfix (cs0 err_src, cs1000 err_src_seed/Pfix_cs1000) and PFw (err_src_seed/PFw_cs{0,1000}); test cells.
d_all = per (origin, channel) squared-error reduction of the carry, hha = |T_t z - z| (rewrite size).
Usage: python -m patchmem.ana_seed --H 192
"""
import argparse
from pathlib import Path

import numpy as np

L = Path('patchmem/logs/server')


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def top_overlap(a, b, f=0.1):
    ta, tb = a >= np.quantile(a, 1 - f), b >= np.quantile(b, 1 - f)
    return float((ta & tb).sum() / ta.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--H', type=int, default=192)
    p.add_argument('--seeds', default='3047,2024,2025')
    a = p.parse_args()
    pairs = {'Pfix': (L / 'err_src', 's1_Pfix', L / 'err_src_seed/Pfix_cs1000', 's1_Pfix'),
             'PFw': (L / 'err_src_seed/PFw_cs0', 's1_PFw', L / 'err_src_seed/PFw_cs1000', 's1_PFw')}
    for s in map(int, a.seeds.split(',')):
        for nm, (d0, a0, d1, a1) in pairs.items():
            x = np.load(d0 / f'T_H{a.H}_s{s}_{a0}_src.npz')
            y = np.load(d1 / f'T_H{a.H}_s{s}_{a1}_src.npz')
            assert (x['rows'] == y['rows']).all()
            te = x['split'] == 2
            dx, dy = x['d_all'][te].astype(np.float64), y['d_all'][te].astype(np.float64)
            hx, hy = x['hha'][te].astype(np.float64), y['hha'][te].astype(np.float64)
            ee = x['ee'][te].astype(np.float64)
            print(f'H{a.H} s{s} {nm:4s} gain {100 * dx.sum() / ee.sum():.3f} / {100 * dy.sum() / ee.sum():.3f} | d_all corr: cell {corr(dx, dy):+.3f} '
                  f'origin {corr(dx.sum(1), dy.sum(1)):+.3f} channel {corr(dx.sum(0), dy.sum(0)):+.3f} top10 overlap {top_overlap(dx, dy):.2f} '
                  f'| hha corr: cell {corr(hx, hy):+.3f} origin {corr(hx.mean(1), hy.mean(1)):+.3f} top10 overlap {top_overlap(hx, hy):.2f}')


if __name__ == '__main__':
    main()
