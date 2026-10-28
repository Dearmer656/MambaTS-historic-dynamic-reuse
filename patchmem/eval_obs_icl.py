"""PAT-315 C group: normalized observation-to-continuation ICL (value = future observations, not backbone errors).

Demo at origin s (age a = t - s >= H, so its continuation is fully observed at t), channel c:
  mu_s, sigma_s   mean / std of y[s-P:s]  (P in {24, 168}; sigma floored at SIG_FLOOR, data is train-standardized)
  k_s = (y[s-R:s] - mu_s) / sigma_s       R = 96, phi = identity
  v_s = (y[s:s+H] - mu_s) / sigma_s
Query t: k_t from y[t-R:t] with mu_t, sigma_t of y[t-P:t].  Predicted normalized future v^ by
  level   v^ = 0                                         (r = mu_t: level-only control)
  B       v^ = sum_j softmax_j(beta k_j.k_t / R) v_j     (retrieval, same dot product; beta = 0 is the plain mean)
  C1      v^ = 1/N sum_j v_j (k_j.k_t / R)               (one W2 GD step from W2 = 0, eta absorbed below)
  C2      v^ = k_t^T (K^T K / N + lam m I)^-1 K^T V / N  (closed-form least-squares W2; m = tr(K^T K / N) / R)
r = mu_t + eta sigma_t v^,  y^ = f + b (r - f),  f = backbone forecast of the L-lookback iTransformer.
Every method gets the same free eta (fitted with b on train), so B / C differ only in how v^ is formed.
Selection: hyper (beta / lam) and eta on train; P and b (global, or per lead) on val; test once.
Demo pools (ages within 2880 - 96 - H, the same for every L):  phase (multiples of 168) | daily (24) | hourly (1).
Statistics are second moments of g1 = mu_t - f and g2 = sigma_t v^ against e = y - f, so any (eta, b) is exact.
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
POOLS = {'phase': 168, 'daily': 24, 'hourly': 1}
BS = {'phase': 2048, 'daily': 512, 'hourly': 64}
SPAN = 2880 - 96                                 # demo ages <= SPAN - H (same window as the residual ICL at L = 96)


def pool_ages(pool):
    step, last = POOLS[pool], SPAN - H
    first = ((H + step - 1) // step) * step
    return list(range(first, last + 1, step))


def methods():
    return [('level', 0.0)] + [('B', b) for b in BETAS] + [('C1', 0.0)] + [('C2', lam) for lam in LAMS]


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


def predict(X, stats, t, c, ages):
    """Per method: v^ (B, H); plus g1 = mu_t - f is built by the caller.  Returns mu_t, sd_t, {method: v^}."""
    mu, sd = stats
    s = t[:, None] - ages[None]                                         # B N
    cc = c[:, None]
    kr, kh = torch.arange(-R, 0, device=X.device), torch.arange(H, device=X.device)
    mu_s, sd_s = mu[s, cc][..., None], sd[s, cc][..., None]
    K = (X[s[..., None] + kr, cc[..., None]] - mu_s) / sd_s            # B N R
    V = (X[s[..., None] + kh, cc[..., None]] - mu_s) / sd_s            # B N H
    mu_t, sd_t = mu[t, c][:, None], sd[t, c][:, None]
    kt = (X[t[:, None] + kr, c[:, None]] - mu_t) / sd_t                 # B R
    dot = (K @ kt[..., None])[..., 0] / R                               # B N
    N = K.shape[1]
    out = {('level', 0.0): torch.zeros(len(t), H, device=X.device)}
    for beta in BETAS:
        out['B', beta] = (torch.softmax(beta * dot, -1)[..., None] * V).sum(1)
    out['C1', 0.0] = (dot[..., None] * V).sum(1) / N
    KK = K.transpose(1, 2).double() @ K.double() / N                    # B R R
    KV = K.transpose(1, 2).double() @ V.double() / N                    # B R H
    m = KK.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
    eye = torch.eye(R, dtype=torch.float64, device=X.device)
    for lam in LAMS:
        W = torch.linalg.solve(KK + (lam * m + 1e-6) * eye, KV)       # B R H; +1e-6: a flat channel gives K = 0
        out['C2', lam] = (kt.double()[:, None] @ W)[:, 0].float()
    return mu_t, sd_t, out


@torch.no_grad()
def accumulate(X, stats_by_p, cases, ages, bs):
    """{(P, method, hyper): (6, H) sums of g1g1, g1g2, g2g2, g1e, g2e, ee}, n cases."""
    t_all, c_all, e_all = cases
    dev = X.device
    ag = torch.as_tensor(ages, device=dev)
    acc = {}
    for i in range(0, len(t_all), bs):
        t, c = t_all[i:i + bs].to(dev), c_all[i:i + bs].to(dev)
        e = e_all[i:i + bs].to(dev).double()
        y = X[t[:, None] + torch.arange(H, device=dev), c[:, None]].double()
        f = y - e
        for P, st in stats_by_p.items():
            mu_t, sd_t, vh = predict(X, st, t, c, ag)
            g1 = mu_t.double() - f
            for key, v in vh.items():
                g2 = sd_t.double() * v.double()
                row = torch.stack([(g1 * g1).sum(0), (g1 * g2).sum(0), (g2 * g2).sum(0), (g1 * e).sum(0),
                                   (g2 * e).sum(0), (e * e).sum(0)])
                k = (P,) + key
                acc[k] = acc[k] + row if k in acc else row
    return {k: v.cpu() for k, v in acc.items()}, len(t_all)


def fit_eta(a):
    """Unconstrained 2-parameter LS of e on (g1, g2) over all leads; eta = c / b (0 if b <= 0 or no g2)."""
    s = a.sum(1)
    if s[2] <= 0:
        return 0.0
    A = torch.tensor([[s[0], s[1]], [s[1], s[2]]], dtype=torch.float64)
    A = A + 1e-9 * float(A.diagonal().sum()) * torch.eye(2, dtype=torch.float64)    # collinear g1, g2
    b, c = torch.linalg.solve(A, torch.stack([s[3], s[4]])).tolist()
    return c / b if b > 1e-12 else 0.0


def dd_ed(a, eta):
    """Per-lead S_dd, S_ed, S_ee of d = g1 + eta g2."""
    return a[0] + 2 * eta * a[1] + eta ** 2 * a[2], a[3] + eta * a[4], a[5]


def fit_b(a, eta, per_lead):
    dd, ed, _ = dd_ed(a, eta)
    if per_lead:
        return (ed / dd.clamp_min(1e-12)).clamp_min(0)
    return torch.full_like(dd, max(0.0, float(ed.sum() / max(float(dd.sum()), 1e-12))))


def mse(a, eta, b, n):
    dd, ed, ee = dd_ed(a, eta)
    return float((ee - 2 * b * ed + b * b * dd).sum()) / (n * H)


def take_cases(data, max_cases, seed):
    t, c = all_cases(data, max_cases, seed)
    e = data['E'][t - L, :, c].clone()
    return t, c, e


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    device = exp.device
    cell = CELLS[code]
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    splits = {}
    for sp, n_cases, cs in (('train', args.select_cases, seed), ('val', args.val_cases, seed),
                            ('test', args.test_cases, seed + 1)):
        data = load_split(exp, sp, cell['history'], args.forecast_bs)
        if sp == 'test':
            test_base = float(data['E'][data['origins'] - L].double().square().mean())
            if abs(test_base - official) > 1e-4:
                raise AssertionError(f'base cached={test_base} official={official}')
        assert int(data['origins'][0]) - (SPAN - H) - max(R, max(PS)) >= 0      # oldest demo window in range
        splits[sp] = dict(X=data['X'].to(device), cases=take_cases(data, n_cases, cs))
        del data
        print(f'[data] {sp} cases {len(splits[sp]["cases"][0])}', flush=True)
    del exp
    torch.cuda.empty_cache()
    stats = {sp: {P: rolling(d['X'], P) for P in PS} for sp, d in splits.items()}
    result = {}
    base = None
    for pool in args.pools.split(','):
        ages = pool_ages(pool)
        acc = {}
        for sp, d in splits.items():
            acc[sp] = accumulate(d['X'], stats[sp], d['cases'], ages, BS[pool])
            print(f'[acc] {pool} {sp} n_demos {len(ages)}', flush=True)
        (a_tr, n_tr), (a_va, n_va), (a_te, n_te) = acc['train'], acc['val'], acc['test']
        base = float(next(iter(a_te.values()))[5].sum()) / (n_te * H)
        for name in ('level', 'B', 'C1', 'C2', 'Bmean'):
            per_p = {}
            for P in PS:
                cands = [m for m in methods() if (m[0] == name or (name == 'Bmean' and m == ('B', 0.0)))]
                best = None
                for m in cands:
                    k = (P,) + m
                    eta = fit_eta(a_tr[k])
                    tr = mse(a_tr[k], eta, fit_b(a_tr[k], eta, False), n_tr)
                    if best is None or tr < best[2]:
                        best = (m[1], eta, tr)
                per_p[P] = dict(hyper=best[0], eta=best[1], train_mse=best[2])
            row = dict(pool=pool, n_demos=len(ages), per_p=per_p)
            for tag, pl in (('global', False), ('lead', True)):
                kp = {P: (P, 'B' if name == 'Bmean' else name, per_p[P]['hyper']) for P in PS}
                vm = {P: mse(a_va[kp[P]], per_p[P]['eta'], fit_b(a_va[kp[P]], per_p[P]['eta'], pl), n_va) for P in PS}
                P = min(vm, key=vm.get)                                 # P chosen on val under this b rule
                x, k = per_p[P], kp[P]
                row[f'P_{tag}'], row[f'val_mse_{tag}'] = P, vm[P]
                row[f'hyper_{tag}'], row[f'eta_{tag}'] = x['hyper'], x['eta']
                b = fit_b(a_va[k], x['eta'], pl)
                m = mse(a_te[k], x['eta'], b, n_te)
                row[f'b_{tag}'] = float(b.mean())
                row[f'test_mse_{tag}'] = m
                row[f'test_gain_{tag}'] = 100 * (base - m) / base
            result[f'{pool}:{name}'] = row
            print(f'[test] {pool:6s} {name:5s} P={row["P_global"]} hyper={row["hyper_global"]} eta={row["eta_global"]:.3g} '
                  f'b={row["b_global"]:.3f} '
                  f'gain={row["test_gain_global"]:+.3f}% lead-b {row["test_gain_lead"]:+.3f}%', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, R=R, Ps=PS, sig_floor=SIG_FLOOR, betas=BETAS, lams=LAMS, span=SPAN,
               checkpoint=str(checkpoint), select_cases=args.select_cases, val_cases=args.val_cases,
               test_cases=args.test_cases, test_base_full=test_base, test_base=base, families=result)
    output = Path(args.out_dir) / f'{code}_H{H}_L{L}_obsicl_s{seed}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T', choices=CELLS)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--pools', default='phase,daily,hourly')
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--select_cases', type=int, default=200000)
    p.add_argument('--val_cases', type=int, default=200000)
    p.add_argument('--test_cases', type=int, default=400000)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
