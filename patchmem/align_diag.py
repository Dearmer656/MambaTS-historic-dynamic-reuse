"""PAT-307 alignment of the learned output corrections with the residual they should fix (seed 3047, jointly trained arms).
For every window (origin t, lookback L, H = 336 = 7 leads of P = 48) of a split, the SAME checkpoint is run twice:
  normal  yhat' = the arm's forecast;   zero  yhat = the arm with its correction set to 0 (set_audit('zero'): R1Obs delta = 0 /
  R1Tefl correction = 0; with feed=True the rollout is fed its own patches, so d also carries the downstream effect).
  r = y - yhat (residual without correction),  d = yhat' - yhat (the correction actually applied),  y - yhat' = r - d.
Per window, lead k, variable c (sums over the P steps of the lead):  a = r'd,  q = |d|^2,  gain = 2a - q = |r|^2 - |r - d|^2.
Reported per split (pooled sums, so they are MSE fractions):
  MSEred = sum gain / sum |r|^2        the fraction of the uncorrected MSE the correction removes (alpha = 1, as deployed)
  pos    = fraction of (window, lead, variable) with a > 0 (direction right);  win = fraction with gain > 0 (net helpful)
  alpha* = sum a / sum q               the best global scale of d on that split (1 = as deployed; < 0.5 = the correction overshoots
                                       so far that alpha = 1 hurts; <= 0 = the direction is wrong on average)
  MSEred* = (sum a)^2 / (sum q sum |r|^2)   what the best global alpha would give on that split
broken down by lead k (and L), by variable, by phase (hour of day of the origin; position of the origin in the split, 10 bins).
Arms: E = ETTh2 (T0 / T0r / O / ErrPath), W = Weather (T0 / O / ErrPath). L in {384, 720, 1440}; train uses every 3rd origin (coprime with the day length).
Figure: patchmem/logs/server/align_{data}.png. usage: python patchmem/align_diag.py --data E|W"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
np.Inf = np.inf
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from patchmem.diag_rollout import official_args
from patchmem.resid_cov import CFG, P, H, NP, SEED, split_x

OBS = dict(R1_OBS='1', R1_MEM_QK_DIM='8', R1_MEM_SLOT_KEEP='0.5')
ARMS = {'E': {'T0': ('R1TfnH0R64h0s3047', dict(R1_TEFL='native', R1_TEFL_HC='0', R1_TEFL_RANK='64')),
              'T0r': ('R1TfnH2R64h0s3047', dict(R1_TEFL='native', R1_TEFL_HC='2', R1_TEFL_RANK='64')),
              'O': ('R1ObK0.5Q8h0s3047', OBS),
              'ErrPath': ('R1ObK0.5Q8PVh0s3047', dict(OBS, R1_OBS_VPATH='1'))},
        'W': {'T0': ('R1WTfnH0R64h0s3047', dict(R1_TEFL='native', R1_TEFL_HC='0', R1_TEFL_RANK='64')),
              'O': ('R1WObK0.5Q8h0s3047', OBS),
              'ErrPath': ('R1WObK0.5Q8PVh0s3047', dict(OBS, R1_OBS_VPATH='1'))}}
ENVK = ('R1_OBS', 'R1_MEM_QK_DIM', 'R1_MEM_SLOT_KEEP', 'R1_OBS_VPATH', 'R1_OBS_VSHARE', 'R1_OBS_VSRC', 'R1_OBS_FORGET', 'R1_OBS_PATH',
        'R1_TEFL', 'R1_TEFL_HC', 'R1_TEFL_RANK', 'R1_FREEZE_FROM')


def border1(data, L, sp, n_rows):
    """Absolute row of data_x[0] (official splits: ETT_hour 12/4/4 months; custom 70/10/20)."""
    if data == 'E':
        return {'train': 0, 'val': 12 * 30 * 24 - L, 'test': 12 * 30 * 24 + 4 * 30 * 24 - L}[sp]
    ntr, nte = int(n_rows * 0.7), int(n_rows * 0.2)
    return {'train': 0, 'val': ntr - L, 'test': n_rows - nte - L}[sp]


@torch.no_grad()
def run(m, X, L, origins, dev, bs=128):
    """Per window, lead, variable: a = r'd, q = |d|^2, rr = |r|^2 (float64 (N, NP, C))."""
    Xt = torch.as_tensor(X)
    A, Q, RR = [], [], []
    for i0 in range(0, len(origins), bs):
        ts = origins[i0:i0 + bs]
        bx = torch.stack([Xt[t - L:t] for t in ts]).to(dev)
        y = torch.stack([Xt[t:t + H] for t in ts]).to(dev)
        m.set_audit(None); p1 = m(bx, NP)['pred']
        m.set_audit('zero'); p0 = m(bx, NP)['pred']; m.set_audit(None)
        r = (y - p0).double().reshape(len(ts), NP, P, -1); d = (p1 - p0).double().reshape(len(ts), NP, P, -1)
        A.append((r * d).sum(2).cpu()); Q.append((d * d).sum(2).cpu()); RR.append((r * r).sum(2).cpu())
    return torch.cat(A).numpy(), torch.cat(Q).numpy(), torch.cat(RR).numpy()


def summ(a, q, rr):
    sa, sq, sr = a.sum(), q.sum(), rr.sum()
    return dict(MSEred=float((2 * sa - sq) / sr), pos=float((a > 0).mean()), win=float((2 * a - q > 0).mean()),
                alpha=float(sa / sq) if sq > 0 else float('nan'), MSEred_opt=float(sa ** 2 / (sq * sr)) if sq > 0 else 0.0,
                sa=float(sa), sq=float(sq), sr=float(sr))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W']); ap.add_argument('--L', default='384,720,1440')
    a_ = ap.parse_args()
    D = a_.data; cf = CFG[D]; dev = 'cuda'
    Ls = [int(v) for v in a_.L.split(',')]
    raw = pd.read_csv(f'dataset/{"ETT-small/ETTh2.csv" if D == "E" else "weather/weather.csv"}')
    n_rows = len(raw); spd = 24 if D == 'E' else 144                                       # steps per day
    from patchmem.exp_r1 import Exp_R1
    res = {}
    for arm, (des, env) in ARMS[D].items():
        for k in [k for k in os.environ if k.startswith('R1_')]:                     # every R1_* knob cleared between arms
            os.environ.pop(k, None)
        os.environ.update(env)
        ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {des} --seed {SEED}').split())
        ex = Exp_R1(ar); m = ex.model
        ck = cf['r1ck'].replace(cf['r1des'], des)
        assert os.path.exists(f'checkpoints/{ck}/checkpoint.pth'), ck
        print(f'[{D} {arm}] checkpoint {ck}', flush=True)
        m.load_state_dict(torch.load(f'checkpoints/{ck}/checkpoint.pth', map_location=dev)); m.eval(); ex._fix_eval_order()
        res[arm] = {}
        for sp in ('train', 'val', 'test'):
            for L in Ls:
                t0 = time.time()
                X = split_x(ar, L, sp)
                origins = list(range(L, len(X) - H + 1, 3 if sp == 'train' else 1))   # 3: coprime with 24 and 144
                a, q, rr = run(m, X, L, origins, dev)
                b1 = border1(D, L, sp, n_rows)
                o = np.asarray(origins)
                hod = ((b1 + o) % spd) * 24 // spd                                          # hour of day of the origin
                pos = np.minimum((o - L) * 10 // max(len(X) - H + 1 - L, 1), 9)             # position in the split, 10 bins
                ent = dict(n=len(origins), all=summ(a, q, rr), lead=[summ(a[:, k], q[:, k], rr[:, k]) for k in range(NP)],
                           var=[summ(a[:, :, c], q[:, :, c], rr[:, :, c]) for c in range(a.shape[2])],
                           hour=[summ(a[hod == h_], q[hod == h_], rr[hod == h_]) for h_ in range(24)],
                           posbin=[summ(a[pos == b_], q[pos == b_], rr[pos == b_]) for b_ in range(10)],
                           sums=dict(a=a.sum((0, 2)).tolist(), q=q.sum((0, 2)).tolist(), rr=rr.sum((0, 2)).tolist()))
                res[arm][f'{sp}_L{L}'] = ent
                s = ent['all']
                print(f'[{D} {arm} {sp} L{L}] n {len(origins)} | MSEred {s["MSEred"]:+.4f} (best alpha {s["alpha"]:.2f} -> {s["MSEred_opt"]:+.4f}) '
                      f'| pos {s["pos"]:.3f} win {s["win"]:.3f} | by lead MSEred ' + ' '.join(f'{e["MSEred"]:+.3f}' for e in ent['lead']) +
                      ' | alpha ' + ' '.join(f'{e["alpha"]:.2f}' for e in ent['lead']) +
                      f' | MSE corrected {(s["sr"] - 2 * s["sa"] + s["sq"]) / rr.size / P:.6f} uncorrected {s["sr"] / rr.size / P:.6f} | {time.time() - t0:.0f}s', flush=True)
        del ex, m; torch.cuda.empty_cache()
        json.dump(res, open(f'patchmem/logs/server/align_{D}_s{SEED}.json', 'w'))
    plot(D, res, Ls)
    print('DONE')


def pooled(res, arm, sp, Ls, key, idx):
    """A breakdown bin pooled over L from its sums: MSEred = sum(2a - q) / sum rr, alpha = sum a / sum q."""
    sa = sum(res[arm][f'{sp}_L{L}'][key][idx]['sa'] for L in Ls); sq = sum(res[arm][f'{sp}_L{L}'][key][idx]['sq'] for L in Ls)
    sr = sum(res[arm][f'{sp}_L{L}'][key][idx]['sr'] for L in Ls)
    return dict(MSEred=(2 * sa - sq) / sr if sr > 0 else float('nan'), alpha=sa / sq if sq > 0 else float('nan'))


def plot(D, res, Ls):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    arms = list(res); name = {'E': 'ETTh2', 'W': 'Weather'}[D]
    spc = {'train': '#1b9e77', 'val': '#7570b3', 'test': '#d95f02'}
    fig, axs = plt.subplots(len(arms), 4, figsize=(18, 3.2 * len(arms)), squeeze=False)
    for r, arm in enumerate(arms):
        for sp in ('train', 'val', 'test'):
            for li, L in enumerate(Ls):
                e = res[arm][f'{sp}_L{L}']
                ls = (':', '--', '-')[li] if len(Ls) == 3 else '-'
                axs[r, 0].plot(range(1, NP + 1), [x['MSEred'] for x in e['lead']], ls, color=spc[sp], marker='o', ms=3, label=f'{sp} L{L}')
                axs[r, 1].plot(range(1, NP + 1), [x['alpha'] for x in e['lead']], ls, color=spc[sp], marker='o', ms=3)
            axs[r, 2].plot(range(24), [pooled(res, arm, sp, Ls, 'hour', h_)['MSEred'] for h_ in range(24)], color=spc[sp], marker='o', ms=3, label=sp)
            axs[r, 3].plot(range(1, 11), [pooled(res, arm, sp, Ls, 'posbin', b_)['MSEred'] for b_ in range(10)], color=spc[sp], marker='o', ms=3, label=sp)
        for c, (ttl, xl) in enumerate((('MSE reduction by lead', 'lead k (48 steps each)'), ('best scale alpha* = sum r.d / sum |d|^2 by lead', 'lead k'),
                                       ('MSE reduction by hour of day (pooled over L)', 'hour of the origin'),
                                       ('MSE reduction by position in the split (pooled over L)', 'decile of the split'))):
            ax = axs[r, c]; ax.axhline(0, color='0.5', lw=0.8); ax.grid(alpha=0.25)
            ax.set_title(f'{arm}: {ttl}', fontsize=9); ax.set_xlabel(xl, fontsize=8)
        axs[r, 1].axhline(1, color='0.5', lw=0.8, ls='--'); axs[r, 1].axhline(0.5, color='0.7', lw=0.8, ls=':')
    axs[0, 0].legend(fontsize=6, ncol=3); axs[0, 2].legend(fontsize=7)
    fig.suptitle(f'{name}: alignment of the learned correction d with the residual r (MSEred = sum(2 r.d - |d|^2) / sum |r|^2), seed 3047', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = f'patchmem/logs/server/align_{D}.png'; fig.savefig(out, dpi=105); plt.close(fig)
    print('saved', out, flush=True)


if __name__ == '__main__':
    main()
