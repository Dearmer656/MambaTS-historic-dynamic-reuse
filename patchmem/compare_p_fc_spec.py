"""PAT-315: P vs FC corrections against the backbone error, per frequency (err_fc_mech merged npz, test, fixed lead).
Cross spectrum along the real time axis, pooled over channels and over log-spaced period bins:
  gain(f)  = |S_rd| / S_rr        (regression amplitude of d on r at f: 1 = right size, 0 = nothing)
  amp(f)   = sqrt(S_dd / S_rr)    (raw spectral amplitude ratio)
  phase(f) = arg S_rd -> lag in hours (> 0: d lags r)
  coh(f)   = |S_rd|^2 / (S_rr S_dd)
  align(f) = share of the total alignment <r, d> coming from the bin
Plus Hilbert instantaneous frequency of d vs r in the daily / weekly bands (weighted by A_r A_d)."""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patchmem.analyze_path_mech import bandpass, BANDS

D = Path('patchmem/logs/server/err_fc_mech')
EDGES = np.array([2, 4, 8, 11, 13, 20, 22.8, 25.2, 28, 40, 60, 80, 100, 140, 160, 176, 196, 400, 3400])
LABELS = [f'{a:g}-{b:g}' for a, b in zip(EDGES[:-1], EDGES[1:])]
METH = ('s1_P', 'FC')
out = {}
fig, ax = plt.subplots(4, 3, figsize=(16, 13))
for s in (3047, 2024, 2025):
    z = np.load(D / f'T_H192_L720_s{s}_mech.npz')
    LH = z['leads'].tolist()
    r = z['r_inc'].astype(np.float64); r -= r.mean(0)
    T = r.shape[0]
    f = np.fft.rfftfreq(T, 1.0); per = np.where(f > 0, 1 / np.maximum(f, 1e-12), np.inf)
    R = np.fft.rfft(r, axis=0)
    bins = np.digitize(per, EDGES) - 1
    for m in METH:
        d = z[f'd_{m}'].astype(np.float64); d -= d.mean(0)
        Dm = np.fft.rfft(d, axis=0)
        for i, h in enumerate(LH):
            srr, sdd = (np.abs(R[:, :, i]) ** 2).sum(1), (np.abs(Dm[:, :, i]) ** 2).sum(1)
            srd = (R[:, :, i] * np.conj(Dm[:, :, i])).sum(1)
            tot = srd.real[1:].sum()
            rows = []
            for b in range(len(LABELS)):
                k = (bins == b) & (f > 0)
                a, dd, c = srr[k].sum(), sdd[k].sum(), srd[k].sum()
                pc = math.sqrt(EDGES[b] * EDGES[b + 1])
                rows.append(dict(band=LABELS[b], gain=abs(c) / a, amp=math.sqrt(dd / a), coh=abs(c) ** 2 / (a * dd),
                                 lag_h=-np.angle(c) / (2 * math.pi) * pc, align=c.real / tot, r_share=a / srr[1:].sum()))
            hil = {}
            for bn, band in BANDS.items():
                ar, ad = hilbert(bandpass(r[:, :, i], 1, *band), axis=0), hilbert(bandpass(d[:, :, i], 1, *band), axis=0)
                fr = np.diff(np.unwrap(np.angle(ar), axis=0), axis=0) / (2 * math.pi)
                fd = np.diff(np.unwrap(np.angle(ad), axis=0), axis=0) / (2 * math.pi)
                w = (np.abs(ar) * np.abs(ad))[1:]
                hil[bn] = dict(per_r_h=1 / ((fr * w).sum() / w.sum()), per_d_h=1 / ((fd * w).sum() / w.sum()),
                               dfreq_cpd_abs=float((np.abs(fd - fr) * w).sum() / w.sum() * 24))
            out[f'{m}|s{s}|h{h}'] = dict(bins=rows, hilbert=hil)
            if s == 3047 and h in (0, 23, 191):
                j = (0, 23, 191).index(h)
                x = np.arange(len(LABELS))
                for row, key, yl in ((0, 'gain', 'regression gain |S_rd|/S_rr'), (1, 'coh', 'coherence'),
                                     (2, 'lag_h', 'lag (h), d after r'), (3, 'align', 'share of <r,d>')):
                    ax[row, j].plot(x, [q[key] for q in rows], 'o-', ms=3, label=m)
                    ax[row, j].set_ylabel(yl); ax[row, j].axhline(0, color='k', lw=.5)
                    ax[row, j].set_xticks(x, LABELS, rotation=60, fontsize=7)
                ax[0, j].set_title(f's3047 lead {h}')
for a in ax.flat:
    a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(D / 'figures' / 'p_vs_fc_spectrum_s3047.png', dpi=90)
(D / 'p_vs_fc_spectrum.json').write_text(json.dumps(out, indent=1, default=float))
keyb = ['11-13', '22.8-25.2', '60-80', '160-176']
for h in (0, 23, 95, 191):
    print(f'== lead {h}')
    for m in METH:
        for b in keyb:
            v = [next(q for q in out[f'{m}|s{s}|h{h}']['bins'] if q['band'] == b) for s in (3047, 2024, 2025)]
            print(f'  {m:5s} {b:>10s}h  r-share {np.mean([q["r_share"] for q in v]):.3f}  gain {"/".join(f"{q["gain"]:.3f}" for q in v)}'
                  f'  amp {"/".join(f"{q["amp"]:.3f}" for q in v)}  coh {"/".join(f"{q["coh"]:.3f}" for q in v)}'
                  f'  lag {"/".join(f"{q["lag_h"]:+.1f}" for q in v)}h  align-share {"/".join(f"{q["align"]:+.2f}" for q in v)}')
        for bn in BANDS:
            v = [out[f'{m}|s{s}|h{h}']['hilbert'][bn] for s in (3047, 2024, 2025)]
            print(f'  {m:5s} hilbert {bn:6s} period r {"/".join(f"{q["per_r_h"]:.1f}" for q in v)}h  d {"/".join(f"{q["per_d_h"]:.1f}" for q in v)}h  |df| {"/".join(f"{q["dfreq_cpd_abs"]:.3f}" for q in v)} cyc/day')
