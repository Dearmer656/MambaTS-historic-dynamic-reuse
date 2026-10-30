"""PAT-315: input-keyed error retrieval (eval_decomp2 term S) stacked on FreshCast.

  y^ = y^_FC + a_b m          y^_FC = f + w^ (r - f) (eval_freshcast, reproduced exactly as in eval_decomp)
  e_FC = y - y^_FC (FreshCast's own error), a_b per lead bucket (NB) by val least squares.
Retrieval as eval_decomp2 S: key = the backbone's input window of the channel (length L, instance-normalised), demos =
same channel, stream origins s <= t - H, top-k by cos(x~_t, x~_s), m[t] = (1 / k) sum_{s in topk} cos(x~_t, x~_s) v_s.
Values: _fc  v = e_FC (FreshCast's error), _e  v = e (the backbone's error, as in eval_decomp2).
v1 arms S_fc / S_e: FreshCast's w^ fixed, y^ = y^_FC + a_b m, a_b per lead bucket (NB) by val LS.
v2 arms J_fc / J_e (primary J_fc): y^ = f + w1 (r - f) + w2 m, global scalars (w1, w2) in [0, 1]^2 by box-constrained
least squares on ALL val origins (FreshCast: w = clip(sum e d / sum d^2, 0, 1) on 100 val origins).  Control FC_refit:
FreshCast's w refitted on all val origins.  k in KS chosen per arm on val.
Criterion (fixed): J_fc beats FreshCast on 3/3 seeds, mean gain >= 0.3%; also reported vs FC_refit.
Stream: origins T0 = n_tr - M_TR .. end, contiguous; train tail and the gaps before val / test are demos only.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_freshcast as fc
import patchmem.eval_decomp as dc
import patchmem.eval_decomp2 as d2
from patchmem.eval_error_path import H, L

M_TR = 2016
NB = dc.NB
KS = (4, 8, 16, 64)
EPS = 1e-12
ARMS = ('S_fc', 'S_e')


def box_ls(a11, a12, a22, b1, b2):
    """argmin over [0, 1]^2 of w^T A w - 2 b^T w (A PSD 2x2): interior solution if feasible, else best edge point."""
    q = lambda w1, w2: a11 * w1 ** 2 + 2 * a12 * w1 * w2 + a22 * w2 ** 2 - 2 * b1 * w1 - 2 * b2 * w2
    cl = lambda x: min(max(x, 0.0), 1.0)
    cands = []
    det = a11 * a22 - a12 ** 2
    if det > 0:
        w1, w2 = (a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det
        if 0 <= w1 <= 1 and 0 <= w2 <= 1:
            return w1, w2
    for w1 in (0.0, 1.0):
        cands.append((w1, cl((b2 - a12 * w1) / a22) if a22 > 0 else 0.0))
    for w2 in (0.0, 1.0):
        cands.append((cl((b1 - a12 * w2) / a11) if a11 > 0 else 0.0, w2))
    return min(cands, key=lambda w: q(*w))


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    if args.smoke:
        fc.N_TRAIN_Q, fc.EPOCHS = 2048, 1
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    fcj = Path(args.out_dir) / f'{code}_H{H}_L{L}_freshcast_s{seed}.json'
    fc_json = json.loads(fcj.read_text())['fc']['test_mse'] if fcj.exists() else None
    Xc, Mc, n_tr, n_va = fc.load_series(exp)
    X = Xc.to(dev)
    ws = fc.win_stats(X)
    sdesc = fc.series_desc(X[:n_tr])
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)).to(dev)
    val_all = torch.arange(n_tr, n_tr + n_va - H + 1)
    val_o = val_all[torch.randperm(len(val_all), generator=torch.Generator().manual_seed(seed))[:fc.VAL_ORIGINS]]
    val_o = val_o.sort().values
    eps, ur = fc.untrained_fit(X, ws, sdesc, order, val_o.to(dev))
    model, rkr_log = fc.train_rkr(X, ws, sdesc, order, n_tr, eps, seed)
    C = X.shape[1] if args.max_c is None else args.max_c
    T0 = n_tr - M_TR
    n = len(X) - H + 1 - T0
    times = torch.arange(T0, T0 + n)
    split = torch.full((n,), -1)                                                  # 0 train tail, 1 val, 2 test
    split[times <= n_tr - H] = 0
    split[(times >= n_tr) & (times <= n_tr + n_va - H)] = 1
    split[times >= n_tr + n_va] = 2
    assert int((split == 2).sum()) == len(X) - H + 1 - n_tr - n_va
    nel = {s: int((split == s).sum()) * C * H for s in (0, 1, 2)}
    is_cal = torch.zeros(n, dtype=torch.bool)
    is_cal[val_o - T0] = True
    kh = torch.arange(H, device=dev)
    bk = dc.lead_bucket(H, NB, dev)

    # ---- stream: backbone error e = y - f and FreshCast's d = r - f (n C H) on the CPU
    E, D = torch.zeros(n, C, H), torch.zeros(n, C, H)
    ex = torch.zeros(4, 3, dtype=torch.float64)                                   # (split 0..2, cal) x (ee, ed, dd)
    with torch.no_grad():
        for i in range(0, n, args.forecast_bs):
            ob = times[i:i + args.forecast_bs]
            Fb = fc.forecast(exp, Xc, Mc, ob)
            for j, t0 in enumerate(ob.tolist()):
                k = i + j
                for c in fc.chunks(C, dev):
                    t = torch.full_like(c, t0)
                    sp = fc.build(X, ws, sdesc, order, t, c)
                    R = fc.readout(X, ws, t, c, sp['s'], sp['p'], sp['val'])
                    pi = model(sp['z'], sp['u'], sp['valid'])
                    r = (pi[..., None] * R).sum(1)
                    f = Fb[j][:, c].T.float()
                    y = X[t[:, None] + kh, c[:, None]]
                    e, d = (y - f).double(), (r - f).double()
                    cc = c.cpu()
                    E[k, cc], D[k, cc] = (y - f).cpu(), (r - f).cpu()
                    row = torch.stack([e.square().sum(), (e * d).sum(), d.square().sum()]).cpu()
                    if split[k] >= 0:
                        ex[split[k]] += row
                    if is_cal[k]:
                        ex[3] += row
            if (i // args.forecast_bs) % 100 == 0:
                print(f'[stream] {i}/{n}', flush=True)
    w_hat = min(max(float(ex[3, 1] / ex[3, 2].clamp_min(EPS)), 0.0), 1.0)
    base = float(ex[2, 0]) / nel[2]
    fc_mse = float(ex[2, 0] - 2 * w_hat * ex[2, 1] + w_hat ** 2 * ex[2, 2]) / nel[2]
    if args.max_c is None:
        if abs(base - official) > 1e-4:
            raise AssertionError(f'base full={base} official={official}')
        if fc_json is not None and abs(fc_mse - fc_json) > 1e-4:
            raise AssertionError(f'fc stream={fc_mse} json={fc_json}')
    print(f'[fc] w_hat {w_hat:.3f} base {base:.5f} fc {fc_mse:.5f} (json {fc_json})', flush=True)
    # ---- retrieval: values [e_FC | e] side by side, all k at once; second moments per (split, bucket) of
    # e (target of the joint fits), d = r - f, m; the e_FC moments follow from e_FC = e - w^ d
    spl = split.to(dev)

    def bsum(x):                                                                 # (b C H) -> (b NB) float64
        return torch.zeros(x.shape[0], NB, dtype=torch.float64, device=dev).index_add_(1, bk, x.double().sum(1))

    def acc(dst, x, ks_):                                                        # dst (3 NB)
        s_ = spl[ks_]
        for q in (0, 1, 2):
            dst[q] += x[s_ == q].sum(0)

    z = lambda: torch.zeros(3, NB, dtype=torch.float64, device=dev)
    mo = {key: z() for key in ('ee', 'ed', 'dd')}
    mk = {(a, k): {key: z() for key in ('em', 'dm', 'mm')} for a in ARMS for k in KS}
    td = times.to(dev)
    with torch.no_grad():
        for c in torch.arange(C).split(d2.CHR):
            cg = c.to(dev)
            Kn = d2.norm_windows(X, td, cg, L)
            Kn = (Kn / Kn.norm(dim=-1, keepdim=True).clamp_min(1e-6)).half()
            Ec, Dc = E[:, c].to(dev), D[:, c].to(dev)
            V = torch.cat([Ec - w_hat * Dc, Ec], -1)                                  # n c 2H
            for i in range(0, n, d2.BQ):
                kq = torch.arange(i, min(n, i + d2.BQ), device=dev)
                ms = d2.topk_read(Kn, V, kq, H, KS)
                e_, d_ = Ec[kq], Dc[kq]
                for key, v in (('ee', e_ * e_), ('ed', e_ * d_), ('dd', d_ * d_)):
                    acc(mo[key], bsum(v), kq)
                for k in KS:
                    for ai, a in enumerate(ARMS):
                        m = ms[k][..., ai * H:(ai + 1) * H]
                        for key, v in (('em', e_ * m), ('dm', d_ * m), ('mm', m * m)):
                            acc(mk[(a, k)][key], bsum(v), kq)
            del Kn, V, Ec, Dc
    ee, ed, dd = mo['ee'], mo['ed'], mo['dd']
    ef2 = ee - 2 * w_hat * ed + w_hat ** 2 * dd                                    # sum e_FC^2
    fc_check = float(ef2[2].sum()) / nel[2]
    assert abs(fc_check - fc_mse) <= 1e-5 * fc_mse, (fc_check, fc_mse)
    g_ = lambda x: 100 * (base - x) / base
    vs = lambda x: 100 * (fc_mse - x) / fc_mse
    res = {}

    # (1) per-bucket a_b on the FC residual, FC's w^ fixed (previous version)
    def fit1(a, k):
        o = mk[(a, k)]
        efm = o['em'] - w_hat * o['dm']                                           # sum e_FC m
        coef = torch.where(o['mm'][1] > 0, efm[1] / torch.where(o['mm'][1] > 0, o['mm'][1], torch.ones_like(o['mm'][1])), torch.zeros_like(o['mm'][1]))
        return coef, (ef2 - 2 * coef * efm + coef ** 2 * o['mm']).sum(-1)
    for a in ARMS:
        grid = {k: fit1(a, k) for k in KS}
        kb = min(KS, key=lambda k: float(grid[k][1][1]))
        coef, sse = grid[kb]
        tm = float(sse[2]) / nel[2]
        res[a] = dict(k=kb, a=coef.tolist(), val_mse=float(sse[1]) / nel[1], test_mse=tm,
                      val_grid={str(k): float(v[1][1]) / nel[1] for k, v in grid.items()})

    # (2) FC's w refitted on all val origins (control) and joint global (w1, w2) in [0, 1]^2 on all val origins
    s_ = lambda x: x.sum(-1)                                                       # (3 NB) -> (3)
    w_re = min(max(float(s_(ed)[1] / s_(dd)[1].clamp_min(EPS)), 0.0), 1.0)
    sse = s_(ee) - 2 * w_re * s_(ed) + w_re ** 2 * s_(dd)
    res['FC_refit'] = dict(w=w_re, val_mse=float(sse[1]) / nel[1], test_mse=float(sse[2]) / nel[2])

    def fit2(a, k):
        o = {key: s_(v) for key, v in mk[(a, k)].items()}
        w1, w2 = box_ls(float(s_(dd)[1]), float(o['dm'][1]), float(o['mm'][1]), float(s_(ed)[1]), float(o['em'][1]))
        sse = (s_(ee) - 2 * w1 * s_(ed) - 2 * w2 * o['em'] + w1 ** 2 * s_(dd) + 2 * w1 * w2 * o['dm']
               + w2 ** 2 * o['mm'])
        return (w1, w2), sse
    for a in ARMS:
        grid = {k: fit2(a, k) for k in KS}
        kb = min(KS, key=lambda k: float(grid[k][1][1]))
        (w1, w2), sse = grid[kb]
        res[f'J_{a[2:]}'] = dict(k=kb, w1=w1, w2=w2, val_mse=float(sse[1]) / nel[1], test_mse=float(sse[2]) / nel[2],
                                 val_grid={str(k): float(v[1][1]) / nel[1] for k, v in grid.items()})
    fc_val = float(ef2[1].sum()) / nel[1]
    for name, v in res.items():
        v['gain'], v['vs_fc'] = g_(v['test_mse']), vs(v['test_mse'])
        print(f'[test] {name:8s} mse {v["test_mse"]:.5f} gain {v["gain"]:+.3f}% vs fc {v["vs_fc"]:+.3f}% '
              f'(val {v["val_mse"]:.5f}, fc val {fc_val:.5f}) '
              + ' '.join(f'{x}={v[x]}' for x in ('k', 'w', 'w1', 'w2') if x in v), flush=True)
    print(f'[test] base {base:.5f} fc {fc_mse:.5f} ({g_(fc_mse):+.3f}%)', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, checkpoint=str(checkpoint), base=base, official=official,
               fc=dict(w=w_hat, test_mse=fc_mse, json_test_mse=fc_json, val_mse=fc_val, gain=g_(fc_mse)),
               rkr_loss=rkr_log, settings=dict(M_TR=M_TR, NB=NB, KS=KS, max_c=args.max_c, smoke=args.smoke), **res)
    suf = '_smoke' if (args.max_c or args.smoke) else ''
    p = Path(args.out_dir) / f'{code}_H{H}_L{L}_fcstack2_s{seed}{suf}.json'
    p.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {p}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T')
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--max_c', type=int, default=None)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
