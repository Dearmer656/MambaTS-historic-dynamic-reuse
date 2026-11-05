"""PAT-315 analysis of eval_sr --src dumps: where the P - Pfix gain sits (per val/test origin x channel cell).
Groupings (hour-of-week, |dmu| / |dls| / P-hha deciles, channel, hours since the channel's last top-10% P event): groups
are ranked by val gain-difference density and taken until <= 20% of cells; reported = their share of the test difference.
Also describes the top-10% P cells (thresholds from val, as in eval_sr --src).
Usage: python -m patchmem.ana_src --H 192
"""
import argparse
import json
from pathlib import Path

import numpy as np

HOUR0_OFF = 4 * 24 + 2          # traffic.csv starts Fri 2016-07-01 02:00 -> hour-of-week, Monday 00:00 = 0
DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def pick(gv, nv, gt, cap=0.2):
    """gv / gt per-group val / test sums, nv per-group val cell counts.  Densest val groups until <= cap of cells."""
    dens = gv / np.maximum(nv, 1)
    order = np.argsort(-dens)
    cum = np.cumsum(nv[order]) / nv.sum()
    sel = order[:max(int((cum <= cap).sum()), 1)]
    return sel, float(gt[sel].sum() / gt.sum()), float(gv[sel].sum() / gv.sum()), float(nv[sel].sum() / nv.sum())


def since_event(ev):
    """ev (rows C) bool, rows in time order -> hours since the last event at or before the row (inf before the first)."""
    out = np.full(ev.shape, np.inf, dtype=np.float32)
    last = np.full(ev.shape[1], -np.inf)
    for r in range(ev.shape[0]):
        last = np.where(ev[r], r, last)
        out[r] = r - last
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--H', type=int, default=192)
    p.add_argument('--dir', default='patchmem/logs/server/err_src')
    p.add_argument('--seeds', default='3047,2024,2025')
    a = p.parse_args()
    out = {}
    for s in map(int, a.seeds.split(',')):
        P = np.load(Path(a.dir) / f'T_H{a.H}_s{s}_s1_P_src.npz')
        F = np.load(Path(a.dir) / f'T_H{a.H}_s{s}_s1_Pfix_src.npz')
        assert (P['rows'] == F['rows']).all()
        sp = P['split']
        va, te = (sp == 1) | (sp == 3), sp == 2
        rows = P['rows']
        assert (np.diff(rows[te]) == 1).all() and (np.diff(rows[va]) == 1).all()
        C = P['d_all'].shape[1]
        ee = P['ee'].astype(np.float64)
        D = {'P': P['d_all'].astype(np.float64), 'Pfix': F['d_all'].astype(np.float64)}
        D['dif'] = D['P'] - D['Pfix']
        r = {'test_gain': {k: float(100 * D[k][te].sum() / ee[te].sum()) for k in D},
             'val_gain': {k: float(100 * D[k][va].sum() / ee[va].sum()) for k in D},
             'test_gain_short': float(100 * (P['d_s'] - F['d_s'])[te].sum() / ee[te].sum())}
        how = (P['times'] + HOUR0_OFF) % 168
        hha = P['hha'].astype(np.float64)
        th = np.quantile(hha[va], 0.9)
        ev = hha >= th
        sinc = np.zeros_like(hha, dtype=np.float32)
        for msk in (va, te):                                                       # val and test streams separately
            sinc[msk] = since_event(ev[msk])
        grp = {
            'how': np.broadcast_to(how[:, None], hha.shape),
            'dmu_dec': None, 'dls_dec': None, 'hha_dec': None,
            'channel': np.broadcast_to(np.arange(C)[None], hha.shape),
            'since_ev': np.digitize(sinc, [1, 6, 24, 72, 168]),                    # 0 | 1-5 | 6-23 | 24-71 | 72-167 | >=168
        }
        for nm, x in (('dmu_dec', np.abs(P['dmu'])), ('dls_dec', np.abs(P['dls'])), ('hha_dec', hha)):
            qs = np.quantile(x[va], np.linspace(0.1, 0.9, 9))
            grp[nm] = np.digitize(x, qs)
        r['groups'] = {}
        for nm, g in grp.items():
            ng = int(g.max()) + 1
            sums = {}
            for zn, msk in (('val', va), ('test', te)):
                gg = g[msk].ravel()
                sums[zn] = {k: np.bincount(gg, D[k][msk].ravel(), ng) for k in ('P', 'dif')}
                sums[zn]['n'] = np.bincount(gg, None, ng)
            res_g = {}
            for k in ('P', 'dif'):
                sel, ts, vs, fr = pick(sums['val'][k], sums['val']['n'], sums['test'][k])
                res_g[k] = dict(sel=sel.tolist()[:20], test_share=ts, val_share=vs, cell_frac=fr)
            if nm in ('since_ev', 'dmu_dec', 'dls_dec', 'hha_dec'):                # full profile (test), % of total
                res_g['profile_test_dif'] = (100 * sums['test']['dif'] / sums['test']['dif'].sum()).tolist()
                res_g['profile_test_P'] = (100 * sums['test']['P'] / sums['test']['P'].sum()).tolist()
                res_g['cells_test'] = (sums['test']['n'] / sums['test']['n'].sum()).tolist()
            if nm == 'how':                                                        # by day x hour, test
                res_g['day_dif'] = [float(100 * sums['test']['dif'][d * 24:(d + 1) * 24].sum() / sums['test']['dif'].sum())
                                    for d in range(7)]
                res_g['hour_dif'] = [float(100 * sums['test']['dif'][h::24].sum() / sums['test']['dif'].sum()) for h in range(24)]
            r['groups'][nm] = res_g
        # top-10% P events: when do they happen, how big is the input change there
        e_te = ev & te[:, None]
        r['events'] = dict(
            frac_test=float(e_te[te].mean()),
            how_hist=np.bincount(np.broadcast_to(how[:, None], hha.shape)[e_te], None, 168).tolist(),
            dmu_abs_ev=float(np.abs(P['dmu'])[e_te].mean()), dmu_abs_all=float(np.abs(P['dmu'])[te].mean()),
            dls_abs_ev=float(np.abs(P['dls'])[e_te].mean()), dls_abs_all=float(np.abs(P['dls'])[te].mean()),
            beta_ev=float(P['beta'][e_te].mean()), beta_all=float(P['beta'][te].mean()),
            hh_ev=float(P['hh'][e_te].mean()), hh_all=float(P['hh'][te].mean()),
            zn_ev=float(P['zn'][e_te].mean()), zn_all=float(P['zn'][te].mean()),
            ucos_ev=float(P['ucos'][e_te].mean()), ucos_all=float(P['ucos'][te].mean()),
            ch_top20_share=float(np.sort(e_te.sum(0))[::-1][:C // 5].sum() / max(e_te.sum(), 1)),
            same_time_frac=float((e_te.sum(1)[te] / C).max()))
        # Pfix events at the same cells? (overlap of top-10% sets)
        hf = F['hha'].astype(np.float64)
        evf = hf >= np.quantile(hf[va], 0.9)
        r['events']['overlap_P_Pfix'] = float((ev & evf)[te].sum() / max(ev[te].sum(), 1))
        out[s] = r
        g = r['groups']
        print(f'== H{a.H} s{s}: test gain P {r["test_gain"]["P"]:+.3f} Pfix {r["test_gain"]["Pfix"]:+.3f} '
              f'dif {r["test_gain"]["dif"]:+.3f} (leads<24 {r["test_gain_short"]:+.3f}) | val dif {r["val_gain"]["dif"]:+.3f}')
        for nm in grp:
            print(f'  [{nm:8s}] dif: top<=20% cells (val-chosen) hold test {g[nm]["dif"]["test_share"]:.2f} '
                  f'(val {g[nm]["dif"]["val_share"]:.2f}, cells {g[nm]["dif"]["cell_frac"]:.2f}) | '
                  f'P: test {g[nm]["P"]["test_share"]:.2f}')
        for nm in ('since_ev', 'hha_dec', 'dmu_dec', 'dls_dec'):
            print(f'  [{nm:8s}] test dif % ' + ' '.join(f'{x:5.1f}' for x in g[nm]['profile_test_dif'])
                  + ' | P % ' + ' '.join(f'{x:5.1f}' for x in g[nm]['profile_test_P'])
                  + ' | cells ' + ' '.join(f'{x:.2f}' for x in g[nm]['cells_test']))
        print('  [day] test dif % ' + ' '.join(f'{d} {x:.1f}' for d, x in zip(DAYS, g['how']['day_dif'])))
        print('  [hour] test dif % ' + ' '.join(f'{x:.1f}' for x in g['how']['hour_dif']))
        e = r['events']
        hh_ = np.array(e['how_hist'])
        top = np.argsort(-hh_)[:6]
        print(f'  [events] top-10% P cells: test frac {e["frac_test"]:.3f}, top hours-of-week '
              + ', '.join(f'{DAYS[t // 24]} {t % 24:02d}h ({hh_[t] / hh_.sum():.1%})' for t in top)
              + f' | |dmu| {e["dmu_abs_ev"]:.2f} vs {e["dmu_abs_all"]:.2f}, |dls| {e["dls_abs_ev"]:.2f} vs {e["dls_abs_all"]:.2f}, '
              f'beta {e["beta_ev"]:.2f} vs {e["beta_all"]:.2f}, hh {e["hh_ev"]:.3f} vs {e["hh_all"]:.3f}, |z| {e["zn_ev"]:.3f} vs '
              f'{e["zn_all"]:.3f}, ucos {e["ucos_ev"]:.2f} vs {e["ucos_all"]:.2f}, top-20% channels hold {e["ch_top20_share"]:.2f}, '
              f'overlap with Pfix events {e["overlap_P_Pfix"]:.2f}')
    Path(a.dir, f'ana_src_H{a.H}.json').write_text(json.dumps({str(k): v for k, v in out.items()}, indent=1) + '\n')


if __name__ == '__main__':
    main()
