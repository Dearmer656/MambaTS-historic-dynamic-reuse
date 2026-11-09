"""PAT-315: put FreshCast / B (eval_freshcast --mech_out) and PaTH (eval_sr --mech) test corrections on one footing.

Everything relative to the backbone (r = E = y - f, d = full correction): P -> base + b P, S_e+P -> base + a S_e + b P,
FC -> w (r_FC - f), B -> y_B - f.  Writes an eval_sr-style mech npz whose 'inc' keys equal the 'tot' keys and whose
'base' is the zero correction, so analyze_path_mech.py / analyze_path_band.py run on it unchanged.
Usage: merge_fc_mech.py <sr mech dir> <fc npz dir> <out dir> [seeds]
"""
import sys
from pathlib import Path
import numpy as np

sr_d, fc_d, out_d = map(Path, sys.argv[1:4])
seeds = sys.argv[4].split(',') if len(sys.argv) > 4 else ['3047', '2024', '2025']
out_d.mkdir(parents=True, exist_ok=True)
for s in seeds:
    z = np.load(sr_d / f'T_H192_L720_s{s}_mech.npz')
    f = np.load(fc_d / f'T_H192_L720_s{s}_fc.npz')
    assert (z['leads'] == f['leads']).all()
    LH = z['leads']
    r_sr = z['r_inc'].astype(np.float64) + z['d_base'].astype(np.float64)        # backbone error at the leads
    r_fc = f['ser_r'].astype(np.float64)
    assert r_sr.shape == r_fc.shape, (r_sr.shape, r_fc.shape)
    err = np.abs(r_sr - r_fc).max() / np.abs(r_fc).max()
    rr = f['rr'].T if f['rr'].shape != z['rr_tot'].shape else f['rr']
    rel = abs(rr.sum() - z['rr_tot'].sum()) / z['rr_tot'].sum()
    print(f's{s}: series max rel diff {err:.2e} (fp16), sum r^2 rel diff {rel:.2e}')
    assert err < 1e-2 and rel < 1e-6
    o = dict(leads=LH, q=z['q'])
    meths = ['base']
    o['base_rd_tot'] = o['base_rd_inc'] = np.zeros_like(rr)
    o['base_dd_tot'] = o['base_dd_inc'] = np.zeros_like(rr)
    for m in ('s1_P', 'S_e+s1_P'):
        for k in ('rd', 'dd'):
            o[f'{m}_{k}_tot'] = o[f'{m}_{k}_inc'] = z[f'{m}_{k}_tot']
        o[f'd_{m}'] = (z[f'd_{m}'].astype(np.float32) + z['d_base'].astype(np.float32)).astype(np.float16)
        meths.append(m)
    for m in ('FC', 'B'):
        for k in ('rd', 'dd'):
            o[f'{m}_{k}_tot'] = o[f'{m}_{k}_inc'] = f[f'{m}_{k}']
        o[f'd_{m}'] = f[f'ser_{m}']
        meths.append(m)
    o['rr_tot'] = o['rr_inc'] = z['rr_tot']
    o['r_inc'] = r_sr.astype(np.float16)
    o['d_base'] = np.zeros_like(o['r_inc'])
    np.savez_compressed(out_d / f'T_H192_L720_s{s}_mech.npz', methods=np.array(meths), **o)
    print(f'  -> {out_d}/T_H192_L720_s{s}_mech.npz methods {meths}')
