"""PAT-315 Step 1: what does the PaTH correction do?  Reads eval_sr --mech npz files ({code}_H{H}_L{L}_s{seed}_mech.npz).

All quantities on the test origins, same samples / leads / channels for every method, standardised units.
  r_tot = E (backbone error), d_tot = base + extra   (vs the backbone)
  r_inc = rho = E - base,     d_inc = extra          (beyond the closed-form base)
1. Exact MSE decomposition per method: gain = [2 mean(r d) - mean(d^2)] / mean(r^2) = alignment term - energy term.
   Also corr rho = <r, d> / (|r| |d|) and relative size s = |d| / |r|: gain = 2 rho s - s^2 <= rho^2 (s = rho optimal),
   i.e. rho = direction / phase agreement, s vs rho = amplitude choice; "amplitude efficiency" = gain / rho^2.
   Per lead and per channel (gain of each (h, c) cell; channel quantiles).
2. Hilbert along the real time axis: for a FIXED lead h and channel c, r_inc(t) and d_inc(t) over consecutive test
   origins t (target t + h), band-passed by an FFT mask to the daily (20-28 h) and weekly (140-196 h) bands, analytic
   signal -> envelope A, phase phi, instantaneous frequency.  Per (h, c): amplitude ratio median(A_d) / median(A_r),
   phase-locking value PLV = |sum w e^(i(phi_d - phi_r))| / sum w (w = A_r A_d), mean phase lag (hours), and the
   band's share of r's energy.  Heatmaps lead x channel (channels sorted by the daily-band energy of r).
3. PaTH / GPhi transition per rotary mode (test, before vs after T_t): mean d log amplitude, mean |d phase|, circular
   mean d phase, energy ratio, and for P the self vs cross-mode part of beta u u^T z.
Diagnostics only: correlations, not causal evidence.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

BANDS = {'daily': (20.0, 28.0), 'weekly': (140.0, 196.0)}


def decomp(z, m, kind):
    rr = z[f'rr_{kind}']
    rd, dd = z[f'{m}_rd_{kind}'], z[f'{m}_dd_{kind}']
    R, D, RD = rr.sum(), dd.sum(), rd.sum()
    rho = RD / np.sqrt(R * D) if D > 0 else 0.0
    s = np.sqrt(D / R)
    gain = (2 * RD - D) / R
    cell = (2 * rd - dd) / np.maximum(rr, 1e-30)                                   # (C H)
    lead = (2 * rd.sum(0) - dd.sum(0)) / rr.sum(0)
    return dict(gain=100 * gain, align=100 * 2 * RD / R, energy=100 * D / R, rho=rho, s=s, opt=100 * rho ** 2,
                eff=gain / rho ** 2 if rho > 0 else 0.0, lead=100 * lead,
                ch_q=np.percentile(100 * (2 * rd.sum(1) - dd.sum(1)) / rr.sum(1), [10, 50, 90]).tolist(),
                ch_pos=float(((2 * rd.sum(1) - dd.sum(1)) > 0).mean()), cell=100 * cell)


def bandpass(x, q, lo_h, hi_h):
    """x (T, ...) along axis 0, keep periods in [lo_h, hi_h] hours."""
    f = np.fft.rfftfreq(x.shape[0], d=1.0 / q)                                       # cycles per hour
    keep = (f >= 1 / hi_h) & (f <= 1 / lo_h)
    X = np.fft.rfft(x, axis=0)
    X[~keep] = 0
    return np.fft.irfft(X, n=x.shape[0], axis=0)


def hilbert_stats(r, d, q, band):
    """r, d (T, C) for one lead.  Returns per-channel amp ratio, PLV, lag (h), band share of r, inst freq of r."""
    rb, db = bandpass(r, q, *band), bandpass(d, q, *band)
    ar, ad = hilbert(rb, axis=0), hilbert(db, axis=0)
    Ar, Ad = np.abs(ar), np.abs(ad)
    dphi = np.angle(ad * np.conj(ar))
    w = Ar * Ad
    plv = np.abs((w * np.exp(1j * dphi)).sum(0)) / np.maximum(w.sum(0), 1e-30)
    mphi = -np.angle((w * np.exp(1j * dphi)).sum(0))                                # > 0: the correction lags r
    fc = 2 / (band[0] + band[1])                                                     # band centre, cycles / h
    lag_h = mphi / (2 * math.pi * fc)
    ifreq = np.median(np.diff(np.unwrap(np.angle(ar), axis=0), axis=0), 0) * q / (2 * math.pi)
    share = (rb ** 2).sum(0) / np.maximum((r - r.mean(0)) ** 2, 1e-30).sum(0)
    amp = np.median(Ad, 0) / np.maximum(np.median(Ar, 0), 1e-12)
    return dict(amp=amp, plv=plv, lag=lag_h, share=share, ifreq_r=ifreq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='patchmem/logs/server/err_sr_mech')
    ap.add_argument('--glob', default='T_H192_L720_s*_mech.npz')
    a = ap.parse_args()
    D = Path(a.dir)
    fig_d = D / 'figures'
    fig_d.mkdir(exist_ok=True)
    summary = {}
    for p in sorted(D.glob(a.glob)):
        z = np.load(p)
        tag = p.name.replace('_mech.npz', '')
        q = float(z['q'])
        LH = z['leads'].tolist()
        meths = z['methods'].tolist()
        S = summary[tag] = {'decomp': {}, 'hilbert': {}, 'modes': {}}
        print(f'== {tag}  (q {q:g}, leads {LH})')
        for kind in ('tot', 'inc'):
            for m in meths:
                if kind == 'inc' and m == 'base':
                    continue
                d_ = decomp(z, m, kind)
                S['decomp'][f'{m}|{kind}'] = {k: v for k, v in d_.items() if k not in ('cell', 'lead')}
                print(f'  [{kind}] {m:12s} gain {d_["gain"]:+.3f}% = align {d_["align"]:.3f} - energy {d_["energy"]:.3f} '
                      f'| rho {d_["rho"]:.3f} s {d_["s"]:.3f} (opt {d_["opt"]:.3f}%, eff {d_["eff"]:.2f}) '
                      f'| channel gain q10/50/90 {"/".join(f"{x:.2f}" for x in d_["ch_q"])} pos {d_["ch_pos"]:.2f}')
        # per-lead gain curves
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
        for j, kind in enumerate(('tot', 'inc')):
            for m in meths:
                if kind == 'inc' and m == 'base':
                    continue
                ax[j].plot(decomp(z, m, kind)['lead'], label=m, lw=1)
            ax[j].axhline(0, color='k', lw=0.5)
            ax[j].set_title(f'{tag}: per-lead gain % ({"vs backbone" if kind == "tot" else "beyond base"})')
            ax[j].set_xlabel('lead')
            ax[j].legend(fontsize=7)
        fig.savefig(fig_d / f'{tag}_lead_gain.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        # Hilbert on the fixed-lead time series (inc: what the extra correction does beyond the base)
        r_inc = z['r_inc'].astype(np.float64)                                         # (T C nL)
        hm = [m for m in meths if m != 'base']
        for band_name, band in BANDS.items():
            stats = {m: [hilbert_stats(r_inc[:, :, i], z[f'd_{m}'][:, :, i].astype(np.float64), q, band)
                         for i in range(len(LH))] for m in hm}
            order = np.argsort(-stats[hm[0]][0]['share'])
            fig, ax = plt.subplots(len(hm), 3, figsize=(16, 2.2 * len(hm)), squeeze=False)
            for i_m, m in enumerate(hm):
                lagl = 6 if band_name == 'daily' else 48
                for j, (key, cm, lim) in enumerate((('plv', 'viridis', (0, 1)), ('amp', 'magma', (0, 1.5)),
                                                     ('lag', 'RdBu_r', (-lagl, lagl)))):
                    M = np.stack([stats[m][i][key][order] for i in range(len(LH))])
                    im = ax[i_m, j].imshow(M, aspect='auto', cmap=cm, vmin=lim[0], vmax=lim[1], interpolation='nearest')
                    ax[i_m, j].set_yticks(range(len(LH)), [f'h{h}' for h in LH])
                    ax[i_m, j].set_title(f'{m}: {key} ({band_name})', fontsize=9)
                    fig.colorbar(im, ax=ax[i_m, j], fraction=0.02)
            ax[-1, 0].set_xlabel('channel (sorted by daily-band share of r)')
            fig.savefig(fig_d / f'{tag}_hilbert_{band_name}.png', dpi=90, bbox_inches='tight')
            plt.close(fig)
            for m in hm:
                for i, h in enumerate(LH):
                    st = stats[m][i]
                    w = st['share']
                    S['hilbert'][f'{m}|{band_name}|h{h}'] = dict(
                        plv_wmean=float((st['plv'] * w).sum() / w.sum()), amp_med=float(np.median(st['amp'])),
                        lag_wmean_h=float((st['lag'] * w).sum() / w.sum()), share_r=float(np.median(st['share'])))
            print(f'  [hilbert {band_name}] ' + ' | '.join(
                f'{m} h{h}: PLV {S["hilbert"][f"{m}|{band_name}|h{h}"]["plv_wmean"]:.2f} '
                f'amp {S["hilbert"][f"{m}|{band_name}|h{h}"]["amp_med"]:.2f}' for m in hm for h in (LH[0], LH[-1])))
        # rotary modes before / after T_t
        arms = sorted({k[len('mode_'):-len('_n')] for k in z.files if k.startswith('mode_') and k.endswith('_n')})
        fig, ax = plt.subplots(1, 3, figsize=(16, 3.5))
        for arm in arms:
            g = lambda key: z[f'mode_{arm}_{key}']
            n = np.maximum(g('n'), 1)
            per = np.where(np.abs(g('theta')) > 0, 2 * math.pi / np.maximum(np.abs(g('theta')), 1e-12) / q, np.inf)
            wn = np.maximum(g('amp2_pre'), 1e-30)                                       # energy-weighted (tiny states noisy)
            dla, dpa = g('w_dlogamp') / wn, g('w_dphase_abs') / wn
            cmean = np.angle(g('dphase_cos') + 1j * g('dphase_sin'))
            er = g('amp2_post') / np.maximum(g('amp2_pre'), 1e-30)
            cross = g('cross2') / np.maximum(g('self2') + g('cross2'), 1e-30)
            S['modes'][arm] = dict(period_h=per.tolist(), dlogamp_w=dla.tolist(), dphase_abs_w=dpa.tolist(),
                                   dlogamp_unw=(g('dlogamp') / n).tolist(), dphase_abs_unw=(g('dphase_abs') / n).tolist(),
                                   dphase_circmean=cmean.tolist(), energy_ratio=er.tolist(), cross_share=cross.tolist())
            o = np.argsort(per)
            ax[0].plot(per[o], dla[o], 'o-', ms=3, label=arm)
            ax[1].plot(per[o], dpa[o], 'o-', ms=3, label=arm)
            if g('cross2').sum() > 0:
                ax[2].plot(per[o], cross[o], 'o-', ms=3, label=arm)
            ew = g('amp2_pre') / g('amp2_pre').sum()
            print(f'  [modes {arm}] energy-weighted d log amp {(dla * ew).sum():+.4f} | |d phase| {(dpa * ew).sum():.4f} rad | '
                  f'energy ratio {er.mean():.4f} | cross-mode share {np.nanmean(cross):.3f}')
        for j, t in enumerate(('mean d log amplitude per step', 'mean |d phase| per step (rad)',
                               'cross-mode share of the Householder update (P)')):
            ax[j].set_xscale('log')
            ax[j].set_xlabel('mode period (h)')
            ax[j].set_title(f'{tag}: {t}', fontsize=9)
            ax[j].legend(fontsize=7)
        fig.savefig(fig_d / f'{tag}_modes.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
    (D / 'mech_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(f'[done] {D / "mech_summary.json"}')


if __name__ == '__main__':
    main()
