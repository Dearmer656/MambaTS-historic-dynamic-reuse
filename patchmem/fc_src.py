"""PAT-316: where does FreshCast's gain on Electricity L720 come from?  Training-free analysis of the FreshCast
reproduction (patchmem/eval_freshcast.py, same frozen iTransformer checkpoint, same RKR recipe and seed).

Question: the residual of a full-lookback backbone is a function of the lookback, so a correction built only from the
lookback has little to add (gain ~ rho^2, rho small).  FreshCast gains ~8% on Electricity L720 H192.  Two hypotheses,
criteria fixed before any result (L720, seeds 3047 / 2024 / 2025, full test set):

  H1 information beyond the lookback.  Each FreshCast candidate continuation is 'in' if all of its data lies inside the
     backbone's lookback [t - L, t) (seasonal: age <= L; shape analog: window end s >= t - L), else 'out'.
     d = w (r - f) = w sum_i pi_i (R_i - f) splits exactly into d_in + d_out.  Restricted arms FC_in / FC_out use the
     same trained scorer with the softmax renormalised over the in / out candidates (no candidate -> no correction),
     each with its own w fitted on the FreshCast validation origins.
     Supported: gain(FC_out) > gain(FC_in) in 3/3 seeds AND alignment share 2<e,d_out> / 2<e,d> >= 0.5 in 3/3.
     Rejected:  gain(FC_in) >= gain(FC_out) in 3/3.
  H2 tracking a drifting backbone bias.  Rm = causal mean of matured backbone errors E[o] (o + H <= t, o >= n_tr) over
     the last W hours, per (lead, channel); y^ = f + a Rm; W in {24, 168, 336, 672, 1344, 2688} and a in [0, 1] fitted
     on the same validation origins.
     Supported: gain(Rm) >= 0.5 gain(FC) in 3/3 AND gain(FC) on the second half of the test origins > first half in 3/3.
     Rejected:  gain(Rm) < 0.25 gain(FC) in 3/3.
  Reported, not criteria: per-half gains of every arm, candidate mass in / out, rho of each correction, joint (FC, Rm)
  least squares on the validation origins (does FC add beyond Rm), L96 as a reference (there almost every candidate is
  'out'), reproduction of the stored FreshCast gain (RKR training is GPU-nondeterministic: warning beyond 0.5 pp).
Caveat: FC_in / FC_out are counterfactual readouts of a scorer trained on all candidates, not separately trained models.

usage: ITR_H=192 ITR_L=720 python -m patchmem.fc_src --data EL --seed 3047 --out_dir patchmem/logs/laptop/fc_src [--smoke]
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_freshcast as fc
from patchmem.eval_error_path import H, L

WINDOWS = (24, 168, 336, 672, 1344, 2688)
ARMS = ('FC', 'FC_in', 'FC_out')


def in_mask(sp, t):
    """True where the candidate continuation lies entirely inside the backbone lookback [t - L, t)."""
    return sp['s'] >= (t - L)[:, None]


def corrections(model, X, ws, sdesc, order, t, c, f):
    """d (B H) for FC / FC_in / FC_out and the exact split pieces of FC: w-free, relative to the forecast f."""
    sp = fc.build(X, ws, sdesc, order, t, c)
    R = fc.readout(X, ws, t, c, sp['s'], sp['p'], sp['val']).double()
    pi = model(sp['z'], sp['u'], sp['valid']).double()
    m = in_mask(sp, t) & sp['valid']
    D = R - f[:, None]                                                          # B N H
    pin, pout = pi * m, pi * (~m)
    part_in, part_out = (pin[..., None] * D).sum(1), (pout[..., None] * D).sum(1)
    min_, mout = pin.sum(1), pout.sum(1)
    d_in = torch.where(min_[:, None] > 0, part_in / min_.clamp_min(1e-300)[:, None], torch.zeros_like(part_in))
    d_out = torch.where(mout[:, None] > 0, part_out / mout.clamp_min(1e-300)[:, None], torch.zeros_like(part_out))
    return dict(FC=part_in + part_out, FC_in=d_in, FC_out=d_out, part_in=part_in, part_out=part_out,
                mass_in=min_, mass_out=mout)


def rm_means(E, o_all, origins, W, c):
    """Causal mean of matured errors E[o] (o + H <= t) over o in [t - H - W, t - H], o in o_all; (n H Cc), 0 if none."""
    Ec = E[:, :, c.cpu()].to(c.device, torch.float64)                            # n_all H Cc
    cs = torch.cat([torch.zeros_like(Ec[:1]), Ec.cumsum(0)])
    o0 = int(o_all[0])
    t = origins.to(c.device)
    hi = (t - H - o0).clamp(-1, len(o_all) - 1)                                  # last matured index (inclusive)
    lo = (t - H - W - o0).clamp(0, len(o_all))
    n = (hi + 1 - lo).clamp_min(0)
    s = cs[hi + 1] - cs[lo.clamp_max(len(o_all))]
    out = s / n.clamp_min(1)[:, None, None]
    return torch.where(n[:, None, None] > 0, out, torch.zeros_like(out))


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    t_start = time.time()
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    Xc, Mc, n_tr, n_va = fc.load_series(exp)
    X = Xc.to(dev)
    C = X.shape[1]
    ws = fc.win_stats(X)
    sdesc = fc.series_desc(X[:n_tr])
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)).to(dev)
    val_o = torch.arange(n_tr, n_tr + n_va - H + 1)                              # identical to eval_freshcast.run
    val_o = val_o[torch.randperm(len(val_o), generator=torch.Generator().manual_seed(seed))[:fc.VAL_ORIGINS]].sort().values
    test_o = torch.arange(n_tr + n_va, len(X) - H + 1)
    if args.smoke:
        test_o = test_o[:: max(1, len(test_o) // 40)]
    eps, _ = fc.untrained_fit(X, ws, sdesc, order, val_o.to(dev))
    model, log = fc.train_rkr(X, ws, sdesc, order, n_tr, eps, seed)
    kh = torch.arange(H, device=dev)

    # backbone errors at every val + test origin (for Rm); CPU float32
    o_all = torch.arange(n_tr, len(X) - H + 1)
    E = torch.zeros(len(o_all), H, C, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(o_all), args.forecast_bs):
            ob = o_all[i:i + args.forecast_bs]
            F = fc.forecast(exp, Xc, Mc, ob)                                     # b H C
            Y = X[(ob.to(dev)[:, None] + kh[None])]                              # b H C
            E[i:i + len(ob)] = (Y - F).float().cpu()
    print(f'[E] {tuple(E.shape)} errors cached ({time.time() - t_start:.0f}s)', flush=True)
    err = lambda o: E[(o - n_tr).long()]                                         # noqa: E731

    # FC / FC_in / FC_out on the validation origins -> w per arm
    with torch.no_grad():
        acc = {a: torch.zeros(2, dtype=torch.float64, device=dev) for a in ARMS}
        dval = {a: torch.zeros(len(val_o), H, C, dtype=torch.float32) for a in ARMS}
        for j, t0 in enumerate(val_o.tolist()):
            e_full = err(torch.tensor([t0]))[0].to(dev).double()               # H C
            for c in fc.chunks(C, dev):
                t = torch.full_like(c, t0)
                e = e_full[:, c].T
                f = X[t[:, None] + kh, c[:, None]].double() - e
                d = corrections(model, X, ws, sdesc, order, t, c, f)
                for a in ARMS:
                    acc[a] += torch.stack([(e * d[a]).sum(), d[a].square().sum()])
                    dval[a][j][:, c.cpu()] = d[a].T.float().cpu()
        w = {a: min(max(float(acc[a][0] / acc[a][1].clamp_min(1e-12)), 0.0), 1.0) for a in ARMS}
    print(f'[val] w {w}', flush=True)

    # Rm: window and coefficient on the validation origins
    e_val = err(val_o)
    rm_val = {}
    for W in WINDOWS:
        rm_val[W] = torch.zeros(len(val_o), H, C, dtype=torch.float32)
        for c in fc.chunks(C, dev):
            rm_val[W][:, :, c.cpu()] = rm_means(E, o_all, val_o, W, c).float().cpu()
    fit = {}
    for W in WINDOWS:
        g = rm_val[W].double()
        a = min(max(float((e_val.double() * g).sum() / g.square().sum().clamp_min(1e-12)), 0.0), 1.0)
        fit[W] = (a, float((e_val.double() - a * g).square().sum()))
    W_rm = min(WINDOWS, key=lambda k: fit[k][1])
    a_rm = fit[W_rm][0]
    print(f'[val] Rm W {W_rm} a {a_rm:.3f}  val sse by W {[round(fit[k][1], 1) for k in WINDOWS]}', flush=True)
    # joint (FC, Rm) on the validation origins: 2x2 least squares, unconstrained, reported only
    g1, g2 = dval['FC'].double(), rm_val[W_rm].double()
    A = torch.tensor([[float((g1 * g1).sum()), float((g1 * g2).sum())], [float((g1 * g2).sum()), float((g2 * g2).sum())]],
                     dtype=torch.float64)
    bvec = torch.tensor([float((e_val.double() * g1).sum()), float((e_val.double() * g2).sum())], dtype=torch.float64)
    wj = torch.linalg.solve(A + 1e-9 * torch.eye(2, dtype=torch.float64), bvec).tolist()
    del dval, rm_val, g1, g2

    # test: every arm, two halves, exact split of FC
    half = len(test_o) // 2
    keys = ('ee', 'FC', 'FC_in', 'FC_out', 'Rm', 'joint', 'ed_FC', 'dd_FC', 'ed_in', 'ed_out', 'dd_in', 'dd_out', 'dd_x',
            'ed_Rm', 'dd_Rm', 'mass_in', 'mass_out', 'n')
    S = [{k: torch.zeros((), dtype=torch.float64, device=dev) for k in keys} for _ in range(2)]
    with torch.no_grad():
        for c in fc.chunks(C, dev):                                              # Rm for the whole test set, this chunk
            rm_c = rm_means(E, o_all, test_o, W_rm, c)                           # nt H Cc (float64, on dev)
            for j, t0 in enumerate(test_o.tolist()):
                hs = S[int(j >= half)]
                t = torch.full_like(c, t0)
                e = E[t0 - n_tr][:, c.cpu()].to(dev).double().T                  # Cc H
                f = X[t[:, None] + kh, c[:, None]].double() - e
                d = corrections(model, X, ws, sdesc, order, t, c, f)
                dfc, drm = w['FC'] * d['FC'], a_rm * rm_c[j].T
                pin, pout = w['FC'] * d['part_in'], w['FC'] * d['part_out']
                hs['ee'] += e.square().sum()
                for a in ARMS:
                    hs[a] += (e - w[a] * d[a]).square().sum()
                hs['Rm'] += (e - drm).square().sum()
                hs['joint'] += (e - wj[0] * d['FC'] - wj[1] * rm_c[j].T).square().sum()
                hs['ed_FC'] += (e * dfc).sum()
                hs['dd_FC'] += dfc.square().sum()
                hs['ed_in'] += (e * pin).sum()
                hs['ed_out'] += (e * pout).sum()
                hs['dd_in'] += pin.square().sum()
                hs['dd_out'] += pout.square().sum()
                hs['dd_x'] += (pin * pout).sum()
                hs['ed_Rm'] += (e * drm).sum()
                hs['dd_Rm'] += drm.square().sum()
                hs['mass_in'] += d['mass_in'].sum()
                hs['mass_out'] += d['mass_out'].sum()
                hs['n'] += e.numel()
            del rm_c
    S = [{k: float(v) for k, v in s.items()} for s in S]
    tot = {k: S[0][k] + S[1][k] for k in S[0]}

    def summ(s):
        ee = s['ee']
        g = {a: 100 * (ee - s[a]) / ee for a in ARMS + ('Rm', 'joint')}
        split = 100 * (2 * s['ed_in'] + 2 * s['ed_out'] - s['dd_in'] - s['dd_out'] - 2 * s['dd_x']) / ee
        rho = lambda ed, dd: ed / max((ee * dd) ** 0.5, 1e-300)                 # noqa: E731
        ncand = s['mass_in'] + s['mass_out']
        return dict(base_mse=ee / s['n'], gain=g, split_gain=split,
                    align_share_out=s['ed_out'] / max(s['ed_in'] + s['ed_out'], 1e-300),
                    rho_FC=rho(s['ed_FC'], s['dd_FC']), rho_Rm=rho(s['ed_Rm'], s['dd_Rm']),
                    mass_in=s['mass_in'] / ncand, mass_out=s['mass_out'] / ncand)
    res = dict(all=summ(tot), first_half=summ(S[0]), second_half=summ(S[1]))
    base = tot['ee'] / tot['n']
    if not args.smoke and abs(base - official) > 1e-4:
        raise AssertionError(f'base full={base} official={official}')
    if abs(res['all']['split_gain'] - res['all']['gain']['FC']) > 1e-6:
        raise AssertionError(f"split {res['all']['split_gain']} != FC {res['all']['gain']['FC']}")
    ref = Path(args.ref_dir) / f'{code}_H{H}_L{L}_freshcast_s{seed}.json'
    ref_gain = json.loads(ref.read_text())['fc']['gain'] if ref.exists() else None
    if ref_gain is not None and abs(ref_gain - res['all']['gain']['FC']) > 0.5:
        print(f'[warn] FC gain {res["all"]["gain"]["FC"]:.3f} vs stored {ref_gain:.3f} (RKR nondeterminism?)', flush=True)
    for k, r in res.items():
        g = r['gain']
        print(f'[test {k:11s}] base {r["base_mse"]:.5f} | gain % FC {g["FC"]:+.3f} FC_in {g["FC_in"]:+.3f} '
              f'FC_out {g["FC_out"]:+.3f} Rm {g["Rm"]:+.3f} joint {g["joint"]:+.3f} | align_out {r["align_share_out"]:.3f} '
              f'mass in/out {r["mass_in"]:.3f}/{r["mass_out"]:.3f} rho FC {r["rho_FC"]:.3f} Rm {r["rho_Rm"]:.3f}', flush=True)
    peak_gpu = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    out = dict(dataset=code, seed=seed, H=H, L=L, checkpoint=str(checkpoint), official=official, base=base,
               smoke=bool(args.smoke), n_test_origins=len(test_o), w=w, rm=dict(W=W_rm, a=a_rm, val_sse=fit),
               joint_w=wj, ref_fc_gain=ref_gain, rkr_loss=log, peak_gpu_gb=peak_gpu, peak_rss_gb=peak_rss,
               seconds=time.time() - t_start, **res)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    p = od / f'{code}_H{H}_L{L}_fcsrc_s{seed}{"_smoke" if args.smoke else ""}.json'
    p.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {p}  peak GPU {peak_gpu:.1f} GB, RSS {peak_rss:.1f} GB, {out["seconds"] / 60:.1f} min', flush=True)


def selftest():
    """CPU checks: rm_means vs brute force; in / out split adds up to FC exactly on random candidates."""
    g = torch.Generator().manual_seed(0)
    o_all = torch.arange(100, 160)
    E = torch.randn(len(o_all), H, 5, generator=g)
    c = torch.arange(1, 4)
    origins = torch.tensor([100, 130, 100 + H, 150 + H, 159 + H + 7])
    for W in (0, 5, 24, 1000):
        got = rm_means(E, o_all, origins, W, c)
        for k, t in enumerate(origins.tolist()):
            sel = [i for i, o in enumerate(o_all.tolist()) if t - H - W <= o <= t - H]
            ref = E[sel][:, :, c].double().mean(0) if sel else torch.zeros(H, len(c), dtype=torch.float64)
            assert torch.allclose(got[k], ref, atol=1e-9), (W, t)
    B, N = 4, 7
    pi = torch.softmax(torch.randn(B, N, generator=g), -1).double()
    R, f = torch.randn(B, N, H, generator=g).double(), torch.randn(B, H, generator=g).double()
    m = torch.rand(B, N, generator=g) > 0.5
    D = R - f[:, None]
    pin, pout = (pi * m)[..., None] * D, (pi * ~m)[..., None] * D
    assert torch.allclose(pin.sum(1) + pout.sum(1), (pi[..., None] * R).sum(1) - f, atol=1e-12)
    sp = dict(s=torch.tensor([[50 - L, 49 - L, 60]]))
    assert in_mask(sp, torch.tensor([50])).tolist() == [[True, False, True]]
    print('SELFTEST OK', flush=True)


def main():
    if '--selftest' in sys.argv:
        return selftest()
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='EL')
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--out_dir', default='patchmem/logs/laptop/fc_src')
    p.add_argument('--ref_dir', default='patchmem/logs/server/error_path')
    p.add_argument('--smoke', action='store_true', help='1 RKR epoch on 2000 queries, ~40 test origins')
    args = p.parse_args()
    if args.smoke:
        fc.N_TRAIN_Q, fc.EPOCHS = 2000, 1
    run(args.data, args.seed, args)


if __name__ == '__main__':
    main()
