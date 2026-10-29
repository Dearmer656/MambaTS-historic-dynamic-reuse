"""PAT-315: case-wise ICL calibration on top of FreshCast (eval_freshcast).

FreshCast combines y^ = f + w^ d, d = r - f, with one global w^ per (dataset, H, forecaster) from 100 validation
origins.  Here each case (t, c) learns its own weight in context from demos = the most recent N fully observed
origins s of the same channel (s <= t - H, so y[s:s+H] is observed at t), each with known e_s = y_s - f_s, d_s:
  w_t = clip((sum_demo e.d + kappa w^ D0_c) / (sum_demo d.d + kappa D0_c), 0, 1)
D0_c = mean per-origin sum d.d of channel c on the FreshCast calibration origins (prior mass of one demo), kappa in
demos.  Forms: scalar (sums over all leads) and lead (NB equal lead buckets, one weight each).  Origins are
streamed contiguously in time order from the validation start to the test end (the H-1 origins between the
splits are demos only); demos before the validation start do not exist
(no demos and kappa = 0 -> w^).  (form, N, kappa) is chosen on all validation origins, test once.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_freshcast as fc
from patchmem.eval_error_path import H, L

NS = (24, 168, 672)
KAPPAS = (0.0, 1.0, 4.0, 16.0, 64.0)
NB = 8


def buckets():
    return torch.arange(H) * NB // H                                             # lead -> bucket


def icl_weights(P, k, Ns, kappas, w_hat, D0):
    """Weights for origin index k from prefix sums P (n+1, 2, NB, C) of (e.d, d.d) per bucket.
    Returns {(form, N, kappa): w (NB, C)}; D0 (NB, C) per-bucket prior mass."""
    out = {}
    hi = k - H + 1                                                               # demos: origin indices <= k - H
    for N in Ns:
        lo = max(hi - N, 0)
        S = P[max(hi, 0)] - P[lo] if hi > 0 else torch.zeros_like(P[0])
        for form in ('scalar', 'lead'):
            ed, dd = (S[0].sum(0, keepdim=True), S[1].sum(0, keepdim=True)) if form == 'scalar' else (S[0], S[1])
            d0 = D0.sum(0, keepdim=True) if form == 'scalar' else D0
            for kap in kappas:
                num, den = ed + kap * w_hat * d0, dd + kap * d0
                w = torch.where(den > 0, num / den.clamp_min(1e-12), torch.full_like(den, w_hat)).clamp(0, 1)
                out[form, N, kap] = w.expand(NB, -1)
    return out


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    Xc, Mc, n_tr, n_va = fc.load_series(exp)
    X = Xc.to(dev)
    C = X.shape[1]
    ws = fc.win_stats(X)
    sdesc = fc.series_desc(X[:n_tr])
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0)).to(dev)
    val_all = torch.arange(n_tr, n_tr + n_va - H + 1)
    val_o = val_all[torch.randperm(len(val_all), generator=torch.Generator().manual_seed(seed))[:fc.VAL_ORIGINS]]
    val_o = val_o.sort().values
    test_o = torch.arange(n_tr + n_va, len(X) - H + 1)
    eps, ur = fc.untrained_fit(X, ws, sdesc, order, val_o.to(dev))
    model, log = fc.train_rkr(X, ws, sdesc, order, n_tr, eps, seed)
    kh = torch.arange(H, device=dev)
    bk = buckets().to(dev)

    def case(F, j, t0):
        """e, d (C, H) for origin t0 with forecasts F[j]."""
        es, dsl = [], []
        for c in fc.chunks(C, dev):
            t = torch.full_like(c, t0)
            sp = fc.build(X, ws, sdesc, order, t, c)
            R = fc.readout(X, ws, t, c, sp['s'], sp['p'], sp['val'])
            r = (model(sp['z'], sp['u'], sp['valid'])[..., None] * R).sum(1)
            y = X[t[:, None] + kh, c[:, None]].double()
            f = F[j][:, c].T.double()
            es.append(y - f)
            dsl.append(r.double() - f)
        return torch.cat(es), torch.cat(dsl)

    def per_bucket(x):                                                           # (C, H) -> (NB, C)
        return torch.zeros(NB, C, dtype=torch.float64, device=dev).index_add_(0, bk, x.T)

    with torch.no_grad():
        ed = dd = 0.0
        d0 = torch.zeros(NB, C, dtype=torch.float64, device=dev)
        for i in range(0, len(val_o), args.forecast_bs):
            ob = val_o[i:i + args.forecast_bs]
            F = fc.forecast(exp, Xc, Mc, ob)
            for j, t0 in enumerate(ob.tolist()):
                e, d = case(F, j, t0)
                ed += float((e * d).sum())
                dd += float(d.square().sum())
                d0 += per_bucket(d.square())
        w_hat = min(max(ed / max(dd, 1e-12), 0.0), 1.0)
        D0 = d0 / len(val_o)
        print(f'[val] w_hat {w_hat:.3f}', flush=True)
        stream = torch.arange(n_tr, len(X) - H + 1)                             # contiguous: index gap == time gap
        t_val_end, t_test = n_tr + n_va - H, n_tr + n_va                         # origins in between: demos only
        P = torch.zeros(len(stream) + 1, 2, NB, C, dtype=torch.float64, device=dev)
        keys = [(fm, N, k) for fm in ('scalar', 'lead') for N in NS for k in KAPPAS]
        sse = {sp: {k: 0.0 for k in keys + ['base', 'fc']} for sp in ('val', 'test')}
        for i in range(0, len(stream), args.forecast_bs):
            ob = stream[i:i + args.forecast_bs]
            F = fc.forecast(exp, Xc, Mc, ob)
            for j, t0 in enumerate(ob.tolist()):
                k = i + j
                sp = 'val' if t0 <= t_val_end else 'test' if t0 >= t_test else None
                e, d = case(F, j, t0)
                if sp is not None:
                    W = icl_weights(P, k, NS, KAPPAS, w_hat, D0)               # uses only demos <= k - H
                    acc = sse[sp]
                    acc['base'] += float(e.square().sum())
                    acc['fc'] += float((e - w_hat * d).square().sum())
                    for key, w in W.items():
                        acc[key] += float((e - w[bk].T * d).square().sum())
                P[k + 1, 0] = P[k, 0] + per_bucket(e * d)
                P[k + 1, 1] = P[k, 1] + per_bucket(d.square())
            if (i // args.forecast_bs) % 50 == 0:
                print(f'[stream] {i}/{len(stream)}', flush=True)
    n = {'val': len(val_all) * C * H, 'test': len(test_o) * C * H}
    m = {sp: {k: v / n[sp] for k, v in sse[sp].items()} for sp in sse}
    base = m['test']['base']
    if abs(base - official) > 1e-4:
        raise AssertionError(f'base full={base} official={official}')
    best = min(keys, key=lambda k: m['val'][k])
    g = lambda x: 100 * (base - x) / base
    res = dict(fc=dict(w=w_hat, test_mse=m['test']['fc'], gain=g(m['test']['fc'])),
               icl=dict(form=best[0], N=best[1], kappa=best[2], val_mse=m['val'][best], val_fc=m['val']['fc'],
                        test_mse=m['test'][best], gain=g(m['test'][best]),
                        vs_fc=100 * (m['test']['fc'] - m['test'][best]) / m['test']['fc']),
               grid={f'{k[0]}_N{k[1]}_k{k[2]}': dict(val=m['val'][k], test=m['test'][k]) for k in keys})
    print(f'[test] base {base:.5f} fc {m["test"]["fc"]:.5f} ({res["fc"]["gain"]:+.3f}%) icl {best} '
          f'{m["test"][best]:.5f} ({res["icl"]["gain"]:+.3f}%, vs fc {res["icl"]["vs_fc"]:+.3f}%)', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, checkpoint=str(checkpoint), base=base, official=official,
               Ns=NS, kappas=KAPPAS, NB=NB, rkr_loss=log, **res)
    p = Path(args.out_dir) / f'{code}_H{H}_L{L}_fcicl_s{seed}.json'
    p.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {p}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T')
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
