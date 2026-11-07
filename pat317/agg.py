"""PAT-317 aggregation: test MSE per (dataset, config, arm, seed) at H 96/192/336/720 from pat_logs/p317, mean over the 4 horizons,
criteria K1 path < rope, K2 path < nope, K3 rope_path < rope (3/3 seeds per dataset; a claim holds iff >= 2/3 datasets), per config.
Usage: python -m pat317.agg [--ep 10]
"""
import argparse
import re
from pathlib import Path

import numpy as np

DS, CFGS, ARMS, SEEDS, HS = ('E1', 'E2', 'W'), ('a', 'b'), ('rope', 'nope', 'path', 'rope_path'), (2021, 2022, 2023), (96, 192, 336, 720)
PAPER = {'E1': (.364, .405, .427, .439), 'W': (.157, .206, .259, .337)}              # Timer-XL Table 10, lookback 672


def read(f):
    out, h = {}, None
    for line in f.read_text(errors='ignore').splitlines():
        m = re.match(r'== H (\d+)', line)
        if m:
            h = int(m.group(1))
        m = re.match(r'mse:([\d.]+)', line)
        if m and h is not None:
            out[h] = float(m.group(1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ep', type=int, default=10)
    a = p.parse_args()
    R = {}
    for D in DS:
        for c in CFGS:
            for arm in ARMS:
                for s in SEEDS:
                    f = Path(f'pat_logs/p317/p317_{D}_{c}_{arm}_s{s}_ep{a.ep}.txt')
                    if f.exists():
                        r = read(f)
                        if all(h in r for h in HS):
                            R[D, c, arm, s] = np.array([r[h] for h in HS])
    for c in CFGS:
        print(f'===== config {c}')
        for D in DS:
            for arm in ARMS:
                v = [R[D, c, arm, s] for s in SEEDS if (D, c, arm, s) in R]
                if not v:
                    continue
                m = np.mean(v, 0)
                print(f'{D} {arm:9s} n={len(v)} H ' + '/'.join(f'{x:.4f}' for x in m) + f' | avg {m.mean():.4f} | per seed '
                      + ' '.join(f'{x.mean():.4f}' for x in v))
            if c == 'a' and D in PAPER:
                print(f'{D} paper     H ' + '/'.join(f'{x:.3f}' for x in PAPER[D]) + f' | avg {np.mean(PAPER[D]):.4f}')
        for k, (x, y) in {'K1 path<rope': ('path', 'rope'), 'K2 path<nope': ('path', 'nope'), 'K3 rope_path<rope': ('rope_path', 'rope')}.items():
            wins, line = 0, []
            for D in DS:
                ok = [(D, c, x, s) in R and (D, c, y, s) in R for s in SEEDS]
                if not all(ok):
                    line.append(f'{D} incomplete')
                    continue
                w = sum(R[D, c, x, s].mean() < R[D, c, y, s].mean() for s in SEEDS)
                d = np.mean([R[D, c, x, s].mean() / R[D, c, y, s].mean() - 1 for s in SEEDS])
                wins += w == 3
                line.append(f'{D} {w}/3 ({100 * d:+.2f}%)')
            print(f'  {k:18s} ' + ', '.join(line) + f' -> datasets 3/3: {wins}/3 ' + ('HOLDS' if wins >= 2 else ''))


if __name__ == '__main__':
    main()
