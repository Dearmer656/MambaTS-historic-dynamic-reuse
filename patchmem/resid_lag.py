"""PAT-307 lag profile of the residual covariance: every single lookback patch j (j = 1 the most recent, up to L/P - 1) against every
rollout lead k = 1..7, per split, for A1 (recurrent R1, no correction) and A0 (official MambaTS), L in {384, 720, 1440}, seed 3047.
Residual definitions as patchmem/resid_cov.py (A1: in-window one-step error of lookback patch j; A0: the first-patch error of A0's
forecast issued at the past origin t - jP; future: y - yhat at lead k; dataset StandardScaler units). Same origins for A1 and A0
per L: t in [2L - P, len - H]. Per variable, then pooled / averaged over variables:
  (a) R2adj(j, k): ridge 48 -> 48 of lead k's residual patch on history patch j, fitted and scored on the same split, adjusted with
      the ridge effective dof (lambda = 0.01 tr(X'X)/48); pooled: 1 - sum_c SSE / sum_c SST.
  (b) corr(j, k): Pearson correlation between the patch-mean residual of history patch j and of lead k, averaged over variables.
Figures: patchmem/logs/server/resid_lag_{data}_{measure}.png (rows: model x L, columns: train / val / test, one line per lead).
usage: python patchmem/resid_lag.py --data E|W"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from patchmem.diag_rollout import official_args
from patchmem.plug_a0 import load_a0, cache_split
from patchmem.resid_cov import CFG, P, H, NP, SEED, split_x, a1_resid, a0_resid


def lag_stats(F, Hs, alpha=0.01):
    """F (n, H, C) future residuals, Hs (n, J, P, C) history patches (j = 1 first). Returns R2adj (J, NP) pooled, corr (J, NP) mean."""
    n, J, _, C = Hs.shape
    sse = np.zeros((J, NP)); sst = np.zeros((J, NP)); dof = np.zeros(J); corr = np.zeros((J, NP))
    for c in range(C):
        Y = F[:, :, c].astype(np.float64); Y = Y - Y.mean(0)
        Yk = Y.reshape(n, NP, P)
        yy = (Yk ** 2).sum((0, 2))
        fm = Yk.mean(2); fm = (fm - fm.mean(0)) / (fm.std(0) + 1e-12)                      # (n, NP)
        for j in range(J):
            X = Hs[:, j, :, c].astype(np.float64); X = X - X.mean(0)
            G = X.T @ X; XY = X.T @ Y
            lam = alpha * np.trace(G) / P
            B = np.linalg.solve(G + lam * np.eye(P), XY)                                   # (P, NP*P)
            fit = (B * XY).sum(0) * 2 - (B * (G @ B)).sum(0)                                # per output column: 2 b'X'y - b'X'Xb
            sse[j] += yy - fit.reshape(NP, P).sum(1)
            sst[j] += yy
            ev = np.linalg.eigvalsh(G); dof[j] += (ev / (ev + lam)).sum() / C
            hm = X.mean(1); hm = (hm - hm.mean()) / (hm.std() + 1e-12)
            corr[j] += (hm[:, None] * fm).mean(0) / C
    r2 = 1 - sse / sst
    return 1 - (1 - r2) * (n - 1) / (n - dof[:, None] - 1), corr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W']); ap.add_argument('--L', default='384,720,1440')
    a_ = ap.parse_args()
    cf = CFG[a_.data]; dev = 'cuda'; res = {}
    os.environ.pop('R1_OBS', None)
    from patchmem.exp_r1 import Exp_R1
    ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {cf["r1des"]} --seed {SEED}').split())
    ex = Exp_R1(ar)
    ex.model.load_state_dict(torch.load(f'checkpoints/{cf["r1ck"]}/checkpoint.pth', map_location=dev))
    for L in [int(v) for v in a_.L.split(',')]:
        t0 = time.time(); J = L // P - 1
        aa = official_args((cf['common'] + ' ' + cf['a0'] + f' --model_id {cf["a0id"].format(L=L)} --seq_len {L} --des ExpA0s{SEED} --seed {SEED}').split())
        a0, _, order = load_a0(aa, cf['a0ck'].format(L=L), SEED)
        for sp in ('train', 'val', 'test'):
            X = split_x(ar, L, sp)
            origins = list(range(L + J * P, len(X) - H + 1))
            Fa, Ha = a1_resid(ex, X, L, origins, dev, mmax=J)
            Y = cache_split(a0, dev, X, L, H, order)[0].numpy()
            Fb, Hb = a0_resid(Y, X, L, origins, mmax=J)
            for mdl, (F, Hs) in (('A1', (Fa, Ha)), ('A0', (Fb, Hb))):
                r2, co = lag_stats(F, Hs)
                res[f'{mdl}_L{L}_{sp}'] = dict(n=len(origins), r2adj=r2.tolist(), corr=co.tolist())
                print(f'[{a_.data} {mdl} L{L} {sp}] n {len(origins)} | R2adj j=1: ' + ' '.join(f'{v:+.3f}' for v in r2[0]) +
                      f' | j={J}: ' + ' '.join(f'{v:+.3f}' for v in r2[-1]) + ' | corr j=1: ' + ' '.join(f'{v:+.3f}' for v in co[0]), flush=True)
        del a0; torch.cuda.empty_cache()
        print(f'[{a_.data} L{L}] {time.time() - t0:.0f}s', flush=True)
        json.dump(res, open(f'patchmem/logs/server/resid_lag_{a_.data}_s{SEED}.json', 'w'))
    plot(a_.data, res, [int(v) for v in a_.L.split(',')])
    print('DONE')


def plot(data, res, Ls):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    name = {'E': 'ETTh2', 'W': 'Weather'}[data]
    cols = plt.cm.viridis(np.linspace(0.0, 0.85, NP))
    for meas, lab in (('r2adj', 'R$^2_{adj}$: history patch j (48) -> lead k (48), ridge'), ('corr', 'corr(patch-mean residual j, lead k)')):
        rows = [(m, L) for m in ('A1', 'A0') for L in Ls]
        fig, axs = plt.subplots(len(rows), 3, figsize=(15, 2.9 * len(rows)), sharey=True, squeeze=False)
        for r, (m, L) in enumerate(rows):
            for c, sp in enumerate(('train', 'val', 'test')):
                ax = axs[r, c]; d = res[f'{m}_L{L}_{sp}']; v = np.asarray(d[meas]); J = v.shape[0]
                for k in range(NP):
                    ax.plot(np.arange(1, J + 1), v[:, k], color=cols[k], lw=1.6, marker='o', ms=2.5, label=f'lead {k + 1}')
                ax.axhline(0, color='0.6', lw=0.8)
                ax.set_title(f'{m}  L={L}  {sp}  (n={d["n"]})', fontsize=10)
                ax.set_xlim(0.5, J + 0.5)
                if r == len(rows) - 1:
                    ax.set_xlabel('lookback patch j (1 = most recent, 48 steps each)')
                if c == 0:
                    ax.set_ylabel(lab if r == 0 else meas, fontsize=8)
                ax.grid(alpha=0.25)
        axs[0, 2].legend(fontsize=7, ncol=2, loc='upper right')
        fig.suptitle(f'{name}: future residual (rollout lead k) vs lookback residual patch j — A1 recurrent R1 / A0 MambaTS, seed 3047', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = f'patchmem/logs/server/resid_lag_{data}_{meas}.png'
        fig.savefig(out, dpi=110); plt.close(fig)
        print('saved', out, flush=True)


if __name__ == '__main__':
    main()
