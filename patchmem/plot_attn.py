"""PAT-304: plots of the rollout retrieval attention recorded by patchmem/mech_attn.py, one row per dataset, for one (L, H):
  (a) window- and head-mean attention mass by slot lag (lag 0 = most recent lookback slot), one line per future patch k; dashed = uniform 1/M
  (b) fraction of (window, head) pairs whose top-1 slot is at that lag, per k
  (c) the same mean attention by slot AGE seen by that patch (age = lag + patch - 1 patches; periodic retrieval lines up here)
  (d) per (window, head) normalised entropy H(a)/log M per k: median and 10-90% band (1 = uniform), effective slots on the right axis
Future patch numbering: the recorded rollout retrieval calls serve future patches 2..H/P; the first future patch is read from the last
history boundary state, whose retrieval happens in the history pass (not recorded here).
usage (repo root): python patchmem/plot_attn.py [--cell 1440/336] [--data E,W,T] [--seed 3047 | 3047,3048,3049] [--out patchmem/logs/mech/attn_1440_336.png]"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAME = {'E': 'ETTh2', 'W': 'Weather (local, 1000 it/epoch)', 'T': 'Traffic-32'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='1440/336'); ap.add_argument('--data', default='E,W,T')
    ap.add_argument('--seed', default='3047', help='one seed or a comma list (one row per data x seed)'); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows = [(d, int(s)) for d in a.data.split(',') for s in a.seed.split(',')]
    fig, ax = plt.subplots(len(rows), 4, figsize=(21, 3.8 * len(rows)), squeeze=False)
    for i, (d, seed) in enumerate(rows):
        r = json.load(open(f'patchmem/logs/mech/attn_{d}_s{seed}.json'))['grid'][a.cell]
        ks = sorted(r, key=int)
        cm = plt.cm.viridis(np.linspace(0, 0.9, len(ks)))
        M = r[ks[0]]['M']
        for c, k in zip(cm, ks):
            fp = int(k) + 2                                                             # future patch served by rollout call k
            ax[i, 0].plot(np.arange(M), r[k]['mass'], color=c, label=f'patch {fp}')
            ax[i, 1].plot(np.arange(M), r[k]['top1_frac'], color=c, label=f'patch {fp}')
            ax[i, 2].plot(np.arange(M) + fp - 1, r[k]['mass'], color=c, label=f'patch {fp}')
        for j in (0, 1):
            ax[i, j].axhline(1 / M, color='gray', ls='--', lw=1, label='uniform 1/M')
            ax[i, j].set_xlabel('slot lag (0 = most recent lookback patch)')
        ax[i, 0].set_ylabel('mean attention'); ax[i, 1].set_ylabel('top-1 fraction')
        ax[i, 0].set_title(f'{NAME[d]} s{seed}  L/H={a.cell}: mean attention by slot lag')
        ax[i, 1].set_title('top-1 slot lag (per window, head)')
        ax[i, 0].legend(fontsize=7, ncol=2)
        ax[i, 2].axhline(1 / M, color='gray', ls='--', lw=1)
        ax[i, 2].set_xlabel('slot age seen by the patch (patches)'); ax[i, 2].set_ylabel('mean attention')
        ax[i, 2].set_title('mean attention by slot age')
        x = np.arange(2, len(ks) + 2)
        q = np.array([r[k]['entropy_q'] for k in ks])
        ax[i, 3].fill_between(x, q[:, 0], q[:, 2], alpha=0.25, label='entropy 10-90%')
        ax[i, 3].plot(x, q[:, 1], 'o-', label='entropy median')
        ax[i, 3].axhline(1, color='gray', ls='--', lw=1)
        ax[i, 3].set_ylim(0, 1.05); ax[i, 3].set_xlabel('future patch'); ax[i, 3].set_ylabel('H(a) / log M')
        tw = ax[i, 3].twinx()
        tw.plot(x, [r[k]['eff_slots_mean'] for k in ks], 's--', color='C3', label='effective slots')
        tw.set_ylabel(f'effective slots (of M={M})', color='C3')
        ax[i, 3].set_title('attention sharpness per future patch')
        ax[i, 3].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    out = a.out or f'patchmem/logs/mech/attn_{a.cell.replace("/", "_")}.png'
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == '__main__':
    main()
