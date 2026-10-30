"""PAT-315: why obs-ICL B beats FreshCast on Traffic L720 -- 2x2 of {candidates} x {combination weights}.

Per (origin t, channel c), e = y - f (backbone error), mu_t / sd_t = B's rolling level / scale (P from the obsicl2 json):
  l    = mu_t - f                      level refresh
  s_fc = r_FC - mu_t                   FreshCast's RKR readout around the level (eval_freshcast)
  s_B  = sd_t v^_B                     B's softmax readout over the same-clock phase pool (eval_freshcast.b_arm)
Arms (all weights global scalars fitted on ALL validation origins, test once):
  FC     f + w (l + s_fc)             w = clip(sum e d / sum d^2, 0, 1)          (FreshCast's form)
  FC_ac  f + a l + c s_fc             (a, c) least squares                        (B's form)
  B_w    f + w (l + s_B)              w clipped as FC
  B      f + a l + c s_B              (a, c) least squares
  FCB    f + a l + c1 s_fc + c2 s_B   least squares (complementarity)
  level  f + a l
References: FC_orig (FreshCast's w^ on its 100 calibration origins), B_orig (the obsicl2 json's a, c).
Criteria (fixed): weighting effect = FC_ac > FC and B > B_w on 3/3 seeds; candidate effect = B > FC_ac and B_w > FC
on 3/3 seeds.  Diagnostics on test: per lead bucket MSE of every arm, per channel share where B beats FC.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_freshcast as fc
import patchmem.eval_obs_icl as oi
from patchmem.eval_error_path import H, L

NB = 8
EPS = 1e-12
RIDGE = 1e-9                                                          # relative, for the LS solves only
NZ3 = 3                                                               # regressors (l, s_fc, s_B)


def lead_bucket(dev):
    return torch.arange(H, device=dev) * NB // H


def solve(G, b, idx):
    """least squares coefficients on the regressor subset idx from Gram G (3 3) and moments b (3)."""
    i = torch.tensor(idx)
    A = G[i][:, i]
    A = A + RIDGE * A.diagonal().mean().clamp_min(EPS) * torch.eye(len(idx), dtype=A.dtype)
    out = torch.zeros(NZ3, dtype=G.dtype)
    out[i] = torch.linalg.solve(A, b[i])
    return out


def single(G, b, idx):
    """FreshCast's clipped single weight on d = sum of the regressors in idx."""
    v = torch.zeros(NZ3, dtype=G.dtype)
    v[list(idx)] = 1.0
    w = float(b @ v / (v @ G @ v).clamp_min(EPS))
    return min(max(w, 0.0), 1.0) * v


def sse(G, b, ee, coef):
    """sum (e - coef . z)^2 from the moments; G (.. 3 3), b (.. 3), ee (..)."""
    return ee - 2 * (b * coef).sum(-1) + torch.einsum('...ij,i,j->...', G, coef, coef)


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    if args.smoke:
        fc.N_TRAIN_Q, fc.EPOCHS = 2048, 1
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    fjs = Path(args.out_dir) / f'{code}_H{H}_L{L}_freshcast_s{seed}.json'
    fc_json = json.loads(fjs.read_text()) if fjs.exists() else None
    Xc, Mc, n_tr, n_va = fc.load_series(exp)
    X = Xc.to(dev)
    C = X.shape[1] if args.max_c is None else args.max_c
    ws = fc.win_stats(X)
    sdesc = fc.series_desc(X[:n_tr])
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)).to(dev)
    val_all = torch.arange(n_tr, n_tr + n_va - H + 1)
    val_o = val_all[torch.randperm(len(val_all), generator=torch.Generator().manual_seed(seed))[:fc.VAL_ORIGINS]]
    val_o = val_o.sort().values
    test_o = torch.arange(n_tr + n_va, len(X) - H + 1)
    eps, ur = fc.untrained_fit(X, ws, sdesc, order, val_o.to(dev))
    model, log = fc.train_rkr(X, ws, sdesc, order, n_tr, eps, seed)
    g = json.loads((Path(args.out_dir) / f'{code}_H{H}_L{L}_obsicl2_s{seed}.json').read_text())
    g = g['families']['phase:B']['global']
    bcfg = dict(P=g['P'], readout=g['readout'], beta=g['hyper'], a=g['a'], c=g['c'])
    ost = oi.rolling(X, bcfg['P'])
    b_ages = torch.as_tensor(oi.pool_cands('phase')[0], device=dev)
    kh = torch.arange(H, device=dev)
    bk = lead_bucket(dev)
    is_cal = torch.zeros(len(X), dtype=torch.bool)
    is_cal[val_o] = True

    def sweep(origins, coefs=None):
        """moments per lead bucket: G (NB 3 3), b (NB 3), ee (NB); for val also the calibration subset's
        FreshCast moments; for test (coefs given) the per-channel SSE of every arm."""
        G = torch.zeros(NB, NZ3, NZ3, dtype=torch.float64, device=dev)
        b = torch.zeros(NB, NZ3, dtype=torch.float64, device=dev)
        ee = torch.zeros(NB, dtype=torch.float64, device=dev)
        cal = torch.zeros(2, dtype=torch.float64, device=dev)                      # sum e d_fc, sum d_fc^2
        chan = None if coefs is None else {k: torch.zeros(C, dtype=torch.float64, device=dev) for k in coefs}
        for i in range(0, len(origins), args.forecast_bs):
            ob = origins[i:i + args.forecast_bs]
            F = fc.forecast(exp, Xc, Mc, ob)                                        # b H C
            for j, t0 in enumerate(ob.tolist()):
                for c in fc.chunks(C, dev):
                    t = torch.full_like(c, t0)
                    sp = fc.build(X, ws, sdesc, order, t, c)
                    R = fc.readout(X, ws, t, c, sp['s'], sp['p'], sp['val'])
                    pi = model(sp['z'], sp['u'], sp['valid'])
                    r = (pi[..., None] * R).sum(1)
                    mt, st, vh = fc.b_arm(X, ost, t, c, b_ages, bcfg['readout'], bcfg['beta'])
                    y = X[t[:, None] + kh, c[:, None]].double()
                    f = F[j][:, c].T.double()
                    mt = mt.double().expand(-1, H)
                    z = torch.stack([mt - f, r.double() - mt, st.double() * vh.double()], -1)   # B' H 3
                    e = y - f
                    G.index_add_(0, bk, torch.einsum('bhi,bhj->hij', z, z))
                    b.index_add_(0, bk, torch.einsum('bhi,bh->hi', z, e))
                    ee.index_add_(0, bk, e.square().sum(0))
                    if is_cal[t0]:
                        d = z[..., 0] + z[..., 1]
                        cal += torch.stack([(e * d).sum(), d.square().sum()])
                    if chan is not None:
                        for k, w in coefs.items():
                            chan[k][c] += (e - z @ w.to(dev)).square().sum(1)
            if (i // args.forecast_bs) % 100 == 0:
                print(f'[sweep] {i}/{len(origins)}', flush=True)
        return G.cpu(), b.cpu(), ee.cpu(), cal.cpu(), None if chan is None else {k: v.cpu() for k, v in chan.items()}

    with torch.no_grad():
        Gv, bv, eev, cal, _ = sweep(val_all)
        G1, b1 = Gv.sum(0), bv.sum(0)
        w_orig = min(max(float(cal[0] / cal[1].clamp_min(EPS)), 0.0), 1.0)
        coefs = {
            'FC_orig': torch.tensor([w_orig, w_orig, 0.0], dtype=torch.float64),
            'B_orig': torch.tensor([bcfg['a'], 0.0, bcfg['c']], dtype=torch.float64),
            'level': solve(G1, b1, (0,)),
            'FC': single(G1, b1, (0, 1)),
            'FC_ac': solve(G1, b1, (0, 1)),
            'B_w': single(G1, b1, (0, 2)),
            'B': solve(G1, b1, (0, 2)),
            'FCB': solve(G1, b1, (0, 1, 2)),
        }
        for k, w in coefs.items():
            print(f'[val] {k:8s} coef {[round(float(x), 4) for x in w]} val mse '
                  f'{float(sse(G1, b1, eev.sum(), w)) / (len(val_all) * C * H):.5f}', flush=True)
        Gt, bt, eet, _, chan = sweep(test_o, coefs)
    n = len(test_o) * C * H
    base = float(eet.sum()) / n
    if args.max_c is None and abs(base - official) > 1e-4:
        raise AssertionError(f'base full={base} official={official}')
    nb = n / NB
    res = {}
    for k, w in coefs.items():
        m = float(sse(Gt.sum(0), bt.sum(0), eet.sum(), w)) / n
        res[k] = dict(coef=w.tolist(), test_mse=m, gain=100 * (base - m) / base,
                      bucket_gain=(100 * (eet - sse(Gt, bt, eet, w)) / eet).tolist(),
                      bucket_mse=(sse(Gt, bt, eet, w) / nb).tolist(),
                      val_mse=float(sse(G1, b1, eev.sum(), w)) / (len(val_all) * C * H))
    for k in res:
        res[k]['vs_fc'] = 100 * (res['FC']['test_mse'] - res[k]['test_mse']) / res['FC']['test_mse']
        print(f'[test] {k:8s} mse {res[k]["test_mse"]:.5f} gain {res[k]["gain"]:+.3f}% vs FC {res[k]["vs_fc"]:+.3f}% '
              f'coef {[round(x, 4) for x in res[k]["coef"]]}', flush=True)
    win = (chan['B'] < chan['FC']).double()
    diag = dict(b_beats_fc_channels=float(win.mean()),
                b_beats_fc_ac_channels=float((chan['B'] < chan['FC_ac']).double().mean()),
                gain_share_top10pct=None)
    dlt = chan['FC'] - chan['B']                                                  # > 0 where B is better
    srt = dlt.sort(descending=True).values
    diag['gain_share_top10pct'] = float(srt[:max(1, C // 10)].sum() / dlt.sum()) if float(dlt.sum()) != 0 else None
    print(f'[diag] {diag}', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, checkpoint=str(checkpoint), base=base, official=official,
               fc_json=None if fc_json is None else dict(fc=fc_json['fc']['gain'], b=fc_json.get('b', {}).get('gain')),
               bcfg=bcfg, n_val_origins=len(val_all), diag=diag, rkr_loss=log, arms=res)
    suf = '_smoke' if (args.max_c or args.smoke) else ''
    p = Path(args.out_dir) / f'{code}_H{H}_L{L}_fcvsb_s{seed}{suf}.json'
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
