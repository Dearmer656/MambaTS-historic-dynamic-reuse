"""PAT-315 C group: normalized observation-to-continuation ICL (value = future observations, not backbone errors).

Candidate (reference origin s, continuation period p), channel c:
  mu_s, sigma_s   mean / std of y[s-P:s]  (P in {24, 168}; sigma floored at SIG_FLOOR, data is train-standardized)
  k_s = (y[s-R:s] - mu_s) / sigma_s       R = 96, phi = identity
  cont_s[h] = y[s + h]            age pools (p = 0): phase / daily / hourly, ages >= H so cont_s is observed at t
  cont_s[h] = y[s + h mod p]      tiled pool: s = t - p j, p in {24, 168}, j = 1..8 (FreshCast seasonal analogs;
                                  only the period after s is used, observed since age >= p)
Two readouts of the continuation, as normalized values v (r = mu_t + sigma_t v):
  shape  v = (cont_s - mu_s) / sigma_s        (re-anchored to the query level and scale)
  value  v = (cont_s - mu_t) / sigma_t        (raw values)
  both   the shape and value candidates of every reference, keys duplicated
Predicted v^:
  level    v^ = 0                                        (r = mu_t)
  naive24 / naive168  value readout of the tile of the last day / week (tiled pool only)
  B        v^ = sum_j softmax_j(beta k_j.k_t / R) v_j     (retrieval, same dot product; Bmean: beta = 0)
  C1       v^ = 1/N sum_j v_j (k_j.k_t / R)               (one W2 GD step from W2 = 0)
  C2       v^ = k_t^T (K^T K / N + lam m I)^-1 K^T V / N  (closed-form least-squares W2; m = tr(K^T K / N) / R)
Combination with decoupled level and readout weights:  y^ = f + a (mu_t - f) + c sigma_t v^.
Selection: readout and hyper (beta / lam) on train (a, c fitted on train); P and (a, c) on val, global or per lead;
test once.  Statistics are second moments of g1 = mu_t - f, g2 = sigma_t v^ and e = y - f, so every fit is exact.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_error_path import CELLS, H, L, all_cases, load_split

R = 96
PS = (24, 168)
SIG_FLOOR = 0.1
BETAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
LAMS = (0.01, 0.1, 1.0, 10.0)
READOUTS = ('shape', 'value', 'both')
POOLS = {'phase': 168, 'daily': 24, 'hourly': 1, 'tiled': None}
TILE_P, TILE_J = (24, 168), 8
BS = {'phase': 2048, 'daily': 256, 'hourly': 24, 'tiled': 2048}
SPAN = 2880 - 96                                 # age-pool ages <= SPAN - H (same window as the residual ICL at L = 96)
FAMILIES = ('level', 'naive24', 'naive168', 'B', 'C1', 'C2', 'Bmean')


def pool_cands(pool):
    """(ages, periods); period 0 = full H-step continuation."""
    if pool == 'tiled':
        c = [(p * j, p) for p in TILE_P for j in range(1, TILE_J + 1)]
        return [a for a, _ in c], [p for _, p in c]
    step, last = POOLS[pool], SPAN - H
    first = ((H + step - 1) // step) * step
    ages = list(range(first, last + 1, step))
    return ages, [0] * len(ages)


def rolling(X, P):
    """mu[i], sigma[i] of X[i-P:i] (rows i < P are NaN)."""
    Xd = X.double()
    z = torch.zeros(1, X.shape[1], dtype=torch.float64, device=X.device)
    c1 = torch.cat([z, Xd.cumsum(0)])
    c2 = torch.cat([z, Xd.square().cumsum(0)])
    mu = torch.full_like(Xd, float('nan'))
    sd = torch.full_like(Xd, float('nan'))
    m = (c1[P:-1] - c1[:-P - 1]) / P
    v = (c2[P:-1] - c2[:-P - 1]) / P - m.square()
    mu[P:] = m
    sd[P:] = v.clamp_min(0).sqrt().clamp_min(SIG_FLOOR)
    return mu.float(), sd.float()


def heads(K, V, kt, dot):
    """B / C1 / C2 predictions from keys K (B N R), values V (B N H)."""
    out = {}
    N = K.shape[1]
    for beta in BETAS:
        out['B', beta] = (torch.softmax(beta * dot, -1)[..., None] * V).sum(1)
    out['C1', 0.0] = (dot[..., None] * V).sum(1) / N
    KK = K.transpose(1, 2).double() @ K.double() / N                    # B R R
    KV = K.transpose(1, 2).double() @ V.double() / N                    # B R H
    m = KK.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
    eye = torch.eye(R, dtype=torch.float64, device=K.device)
    for lam in LAMS:
        W = torch.linalg.solve(KK + (lam * m + 1e-6) * eye, KV)       # B R H; +1e-6: a flat channel gives K = 0
        out['C2', lam] = (kt.double()[:, None] @ W)[:, 0].float()
    return out


def predict(X, stats, t, c, ages, periods):
    """Returns mu_t, sd_t (B, 1) and {(family, readout, hyper): v^ (B, H)}."""
    mu, sd = stats
    dev = X.device
    s = t[:, None] - ages[None]                                         # B N
    cc = c[:, None]
    kr, kh = torch.arange(-R, 0, device=dev), torch.arange(H, device=dev)
    off = torch.where(periods[:, None] > 0, kh[None] % periods.clamp_min(1)[:, None], kh[None])     # N H
    mu_s, sd_s = mu[s, cc][..., None], sd[s, cc][..., None]
    K = (X[s[..., None] + kr, cc[..., None]] - mu_s) / sd_s            # B N R
    cont = X[s[..., None] + off[None], cc[..., None]]                   # B N H
    mu_t, sd_t = mu[t, c][:, None], sd[t, c][:, None]
    kt = (X[t[:, None] + kr, c[:, None]] - mu_t) / sd_t                 # B R
    vals = {'shape': (cont - mu_s) / sd_s, 'value': (cont - mu_t[..., None]) / sd_t[..., None]}
    out = {('level', '-', 0.0): torch.zeros(len(t), H, device=dev)}
    for p in TILE_P:
        hit = ((ages == p) & (periods == p)).nonzero()
        if len(hit):
            out[f'naive{p}', 'value', 0.0] = vals['value'][:, int(hit[0])]
    dot = (K @ kt[..., None])[..., 0] / R                               # B N
    for ro in READOUTS:
        if ro == 'both':
            Kr, Vr, dr = torch.cat([K, K], 1), torch.cat([vals['shape'], vals['value']], 1), torch.cat([dot, dot], 1)
        else:
            Kr, Vr, dr = K, vals[ro], dot
        for (fam, hyper), v in heads(Kr, Vr, kt, dr).items():
            out[fam, ro, hyper] = v
    return mu_t, sd_t, out


@torch.no_grad()
def accumulate(X, stats_by_p, cases, ages, periods, bs):
    """{(P, family, readout, hyper): (6, H) per-lead sums of g1g1, g1g2, g2g2, g1e, g2e, ee}, n cases."""
    t_all, c_all, e_all = cases
    dev = X.device
    ag, pe = torch.as_tensor(ages, device=dev), torch.as_tensor(periods, device=dev)
    acc = {}
    for i in range(0, len(t_all), bs):
        t, c = t_all[i:i + bs].to(dev), c_all[i:i + bs].to(dev)
        e = e_all[i:i + bs].to(dev).double()
        y = X[t[:, None] + torch.arange(H, device=dev), c[:, None]].double()
        f = y - e
        for P, st in stats_by_p.items():
            mu_t, sd_t, vh = predict(X, st, t, c, ag, pe)
            g1 = mu_t.double() - f
            for key, v in vh.items():
                g2 = sd_t.double() * v.double()
                row = torch.stack([(g1 * g1).sum(0), (g1 * g2).sum(0), (g2 * g2).sum(0), (g1 * e).sum(0),
                                   (g2 * e).sum(0), (e * e).sum(0)])
                k = (P,) + key
                acc[k] = acc[k] + row if k in acc else row
    return {k: v.cpu() for k, v in acc.items()}, len(t_all)


def fit_ac(a, per_lead):
    """LS weights (a, c) of e on (g1, g2), per lead (H,) or one global pair broadcast; tiny ridge for g2 = 0."""
    s = a if per_lead else a.sum(1, keepdim=True)
    A = torch.stack([torch.stack([s[0], s[1]], -1), torch.stack([s[1], s[2]], -1)], -2)          # (H|1) 2 2
    A = A + 1e-9 * (s[0] + s[2]).clamp_min(1e-12)[:, None, None] * torch.eye(2, dtype=torch.float64)
    w = torch.linalg.solve(A, torch.stack([s[3], s[4]], -1))                                      # (H|1) 2
    return w[:, 0].expand(a.shape[1]), w[:, 1].expand(a.shape[1])


def mse_ac(a, wa, wc, n):
    sse = a[5] - 2 * (wa * a[3] + wc * a[4]) + wa * wa * a[0] + 2 * wa * wc * a[1] + wc * wc * a[2]
    return float(sse.sum()) / (n * H)


def family_keys(keys, P, name):
    base, _, ro = name.partition(':')
    if base == 'Bmean':
        out = [k for k in keys if k[:2] == (P, 'B') and k[3] == 0.0]
    else:
        out = [k for k in keys if k[:2] == (P, base)]
    return [k for k in out if not ro or k[2] == ro]


def take_cases(data, max_cases, seed):
    t, c = all_cases(data, max_cases, seed)
    e = data['E'][t - L, :, c].clone()
    return t, c, e


def select(acc, n_cands):
    (a_tr, n_tr), (a_va, n_va), (a_te, n_te) = acc['train'], acc['val'], acc['test']
    base = float(next(iter(a_te.values()))[5].sum()) / (n_te * H)
    names = list(FAMILIES) + [f'{m}:{ro}' for m in ('B', 'C1', 'C2', 'Bmean') for ro in READOUTS]
    result = {}
    for name in names:
        per_p = {}
        for P in PS:
            keys = family_keys(a_tr, P, name)
            if not keys:
                continue
            tr = {k: mse_ac(a_tr[k], *fit_ac(a_tr[k], False), n_tr) for k in keys}
            best = min(tr, key=tr.get)
            per_p[P] = dict(key=best, train_mse=tr[best])
        if not per_p:
            continue
        row = dict(n_cands=n_cands, train_mse={P: x['train_mse'] for P, x in per_p.items()})
        for tag, pl in (('global', False), ('lead', True)):
            vm = {P: mse_ac(a_va[x['key']], *fit_ac(a_va[x['key']], pl), n_va) for P, x in per_p.items()}
            P = min(vm, key=vm.get)                                     # P chosen on val under this weight rule
            k = per_p[P]['key']
            wa, wc = fit_ac(a_va[k], pl)
            m = mse_ac(a_te[k], wa, wc, n_te)
            row[tag] = dict(P=P, readout=k[2], hyper=k[3], val_mse=vm[P], a=float(wa.mean()), c=float(wc.mean()),
                            test_mse=m, test_gain=100 * (base - m) / base)
        result[name] = row
    return result, base


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    device = exp.device
    cell = CELLS[code]
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    oldest = max(SPAN - H, max(TILE_P) * TILE_J)
    splits = {}
    for sp, n_cases, cs in (('train', args.select_cases, seed), ('val', args.val_cases, seed),
                            ('test', args.test_cases, seed + 1)):
        data = load_split(exp, sp, cell['history'], args.forecast_bs)
        if sp == 'test':
            test_base = float(data['E'][data['origins'] - L].double().square().mean())
            if abs(test_base - official) > 1e-4:
                raise AssertionError(f'base cached={test_base} official={official}')
        assert int(data['origins'][0]) - oldest - max(R, max(PS)) >= 0          # oldest candidate window in range
        splits[sp] = dict(X=data['X'].to(device), cases=take_cases(data, n_cases, cs))
        del data
        print(f'[data] {sp} cases {len(splits[sp]["cases"][0])}', flush=True)
    del exp
    torch.cuda.empty_cache()
    stats = {sp: {P: rolling(d['X'], P) for P in PS} for sp, d in splits.items()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result, bases, raw = {}, {}, {}
    for pool in args.pools.split(','):
        ages, periods = pool_cands(pool)
        acc = {}
        for sp, d in splits.items():
            acc[sp] = accumulate(d['X'], stats[sp], d['cases'], ages, periods, BS[pool])
            print(f'[acc] {pool} {sp} n_cands {len(ages)}', flush=True)
        raw[pool] = acc
        res, bases[pool] = select(acc, len(ages))
        for name, row in res.items():
            result[f'{pool}:{name}'] = row
            g, l = row['global'], row['lead']
            print(f'[test] {pool:6s} {name:14s} P={g["P"]} ro={g["readout"]} hyper={g["hyper"]} a={g["a"]:.3f} '
                  f'c={g["c"]:.3f} gain={g["test_gain"]:+.3f}% lead {l["test_gain"]:+.3f}%', flush=True)
    stem = f'{code}_H{H}_L{L}_obsicl2_s{seed}'
    torch.save(raw, out_dir / f'{stem}_stats.pt')
    out = dict(dataset=code, seed=seed, H=H, L=L, R=R, Ps=PS, sig_floor=SIG_FLOOR, betas=BETAS, lams=LAMS, span=SPAN,
               tile=(TILE_P, TILE_J), checkpoint=str(checkpoint), select_cases=args.select_cases,
               val_cases=args.val_cases, test_cases=args.test_cases, test_base_full=test_base, test_base=bases,
               families=result)
    (out_dir / f'{stem}.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {out_dir / stem}.json', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T', choices=CELLS)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--pools', default='tiled,phase,daily')
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--select_cases', type=int, default=200000)
    p.add_argument('--val_cases', type=int, default=200000)
    p.add_argument('--test_cases', type=int, default=400000)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
