"""PAT-315: what do the errors that every carry seed corrects have in common?  (descriptive, eval_sr --src dumps, test cells)
S = test (origin, channel) cells in the top 10% of d_all for both carry seeds (cs0, cs1000), per arm (Pfix, PFw).
(S1) is S just the large base errors?  top-10% d_all overlap cs0 vs cs1000 within each decile of ee (remaining error after
     base_sh; deciles from val), mean over deciles; chance .10.
(S2) per-origin corr of the relative gain sum d / sum ee, cs0 vs cs1000.
Descriptive: S vs all test cells (ee, |dmu|, |dls|, hour-of-day / day profile, channel concentration, run length over
consecutive origins of one channel), share of the test gain in S, and the P - Pfix test difference (cs0) in Pfix's S.
Usage: python -m patchmem.ana_stable --H 192
"""
import argparse
from pathlib import Path

import numpy as np

L = Path('patchmem/logs/server')
HOUR0_OFF = 4 * 24 + 2
DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def top(a, f=0.1):
    return a >= np.quantile(a, 1 - f)


def mean_run(m):
    """m (rows C) bool, rows consecutive -> mean length of the runs of True along rows (per channel)."""
    runs = []
    for c in range(m.shape[1]):
        x = np.concatenate([[0], m[:, c].astype(np.int8), [0]])
        d = np.diff(x)
        runs.append(np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0])
    r = np.concatenate(runs)
    return float(r.mean()) if len(r) else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--H', type=int, default=192)
    p.add_argument('--seeds', default='3047,2024,2025')
    a = p.parse_args()
    pairs = {'Pfix': (L / 'err_src', L / 'err_src_seed/Pfix_cs1000', 's1_Pfix'),
             'PFw': (L / 'err_src_seed/PFw_cs0', L / 'err_src_seed/PFw_cs1000', 's1_PFw')}
    rng = np.random.default_rng(0)
    for s in map(int, a.seeds.split(',')):
        P = np.load(L / 'err_src' / f'T_H{a.H}_s{s}_s1_P_src.npz')
        for nm, (d0, d1, arm) in pairs.items():
            x = np.load(d0 / f'T_H{a.H}_s{s}_{arm}_src.npz')
            y = np.load(d1 / f'T_H{a.H}_s{s}_{arm}_src.npz')
            assert (x['rows'] == y['rows']).all() and (x['rows'] == P['rows']).all()
            sp = x['split']
            va, te = (sp == 1) | (sp == 3), sp == 2
            assert (np.diff(x['rows'][te]) == 1).all()
            dx, dy = x['d_all'][te].astype(np.float64), y['d_all'][te].astype(np.float64)
            ee = x['ee'][te].astype(np.float64)
            assert np.allclose(ee, y['ee'][te])
            tx, ty = top(dx), top(dy)
            S = tx & ty
            # S1: overlap within ee deciles (val edges)
            qs = np.quantile(x['ee'][va], np.linspace(0.1, 0.9, 9))
            dec = np.digitize(ee, qs)
            ov = []
            for k in range(10):
                mk = dec == k
                if mk.sum() < 100:
                    continue
                a_, b_ = top(dx[mk]), top(dy[mk])
                ov.append((a_ & b_).sum() / a_.sum())
            ov_dec = float(np.mean(ov))
            # S2: relative gain per origin
            rel = corr(dx.sum(1) / ee.sum(1), dy.sum(1) / ee.sum(1))
            # descriptive
            gx, gy = dx.sum(), dy.sum()
            dm, dl = np.abs(x['dmu'][te]), np.abs(x['dls'][te])
            how = (x['times'][te] + HOUR0_OFF) % 168
            hb = np.broadcast_to(how[:, None], S.shape)
            hod_S = np.bincount(hb[S] % 24, None, 24) / S.sum()
            day_S = np.bincount(hb[S] // 24, None, 7) / S.sum()
            ch_S = np.sort(S.sum(0))[::-1]
            C = S.shape[1]
            ee_rank = (ee[S][:, None] > np.sort(ee.ravel())[None, ::max(ee.size // 1000, 1)]).mean()
            Rm = rng.random(S.shape) < S.mean()
            pd_ = (P['d_all'][te] - x['d_all'][te]).astype(np.float64) if nm == 'Pfix' else None
            print(f'H{a.H} s{s} {nm:4s} | S frac {S.mean():.3f} | S1 overlap within ee deciles {ov_dec:.2f} (all {S.sum() / tx.sum():.2f}) '
                  f'| S2 rel-gain origin corr {rel:+.2f} | S holds gain {dx[S].sum() / gx:.2f}/{dy[S].sum() / gy:.2f}, '
                  f'top10 holds {dx[tx].sum() / gx:.2f}, negative-gain cells hold {dx[dx < 0].sum() / gx:+.2f}')
            print(f'    S vs all: median ee {np.median(ee[S]) / np.median(ee):.1f}x, mean ee pct {ee_rank:.2f}, share of S in top-10% ee {(top(ee) & S).sum() / S.sum():.2f}, '
                  f'rel gain d/ee in S {dx[S].sum() / ee[S].sum():.3f} vs all {gx / ee.sum():.4f} | |dmu| {np.median(dm[S]) / np.median(dm):.2f}x '
                  f'|dls| {np.median(dl[S]) / np.median(dl):.2f}x | top-20% channels hold {ch_S[:C // 5].sum() / S.sum():.2f} '
                  f'| mean run (h) {mean_run(S):.1f} vs random {mean_run(Rm):.1f}, top10-ee {mean_run(top(ee)):.1f}')
            print('    S hour-of-day % ' + ' '.join(f'{100 * v:.0f}' for v in hod_S)
                  + ' | day % ' + ' '.join(f'{d} {100 * v:.0f}' for d, v in zip(DAYS, day_S)))
            if pd_ is not None:
                print(f'    P - Pfix (cs0) test diff {100 * pd_.sum() / ee.sum():+.3f}%: share in Pfix S {pd_[S].sum() / pd_.sum():.2f}, '
                      f'in top-10% ee {pd_[top(ee)].sum() / pd_.sum():.2f}, in P top-10% d {pd_[top(P["d_all"][te].astype(np.float64))].sum() / pd_.sum():.2f}')


if __name__ == '__main__':
    main()
