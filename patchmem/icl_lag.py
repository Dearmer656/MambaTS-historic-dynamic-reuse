"""PAT-307 offline multi-lag ICL probe (round 2 of patchmem/icl_probe.py; no model training; A1 / A0 seed-3047 checkpoints, frozen).
Question: round 1 (adjacent-lag demos) found in-window pairing only at leads 1-3; at leads 4-7 icl_ridge ~ shuffle (bias only).
Is that because the inputs were only the patches ADJACENT to lag k? Here a demonstration may read any lag set G (non-adjacent
intervals and near+far combinations), every lag is offered (no lag is hand-picked from what the test split showed), and each
window decides via leave-one-out (LOO) on its own past errors which input set to trust, including abstaining. This is the online /
in-context protocol: the window's past (already observed) residuals are used at prediction time; the future target never is.
Residuals as resid_cov.py / icl_probe.py: chronological in-window errors e_0..e_{J-1} (J = L/P - 1, e_{J-1} most recent), target
r_k = the rollout residual of lead k (position J-1+k). phi(patch) = (mean, LS slope) per variable.
Demonstration for lead k with lag set G (all g >= k, so the query is observed): input = [phi(e_{u-g})]_{g in G}, output = e_u, for
u = max(G)..J-1 (n = J - max(G) demos, required n >= NMIN); query input = [phi(e_{J-1+k-g})]_{g in G}.
Solver (all families): kernel ridge in dual form with an unpenalised intercept, features standardised on the window's demos:
pred = ybar + kc(q)^T (Kc + lam' I)^-1 (Y - ybar), lam' = lam * tr(Kc)/n (floored), exact LOO residual (y - yhat)/(1 - S_ii),
S = 11^T/n + Kc (Kc + lam' I)^-1 (exact for the fixed smoother: the standardisation and RBF bandwidth are not re-estimated per
left-out row). Linear kernel or RBF exp(-d^2 / (h^2 median d^2)) over the demos.
Candidate families (each a fixed config; per variable unless 'pool' = one map shared by all variables of the window):
  adj    G = {k..k+m-1}, m in {1,2,3}, linear (round-1 inputs)
  lin    G = {k..k+S-1}, S in SPANS, linear, pool in {0,1}          (all intervals up to k+S-1)
  rbf    G = {k..k+S-1}, S in RSPANS, RBF (lag combinations / interactions), pool in {0,1}
  xvar   G = {k..k+S-1}, features of ALL variables at those lags, linear, per target variable
  single G = {g}, every g in k..J-NMIN, linear lam=1 (one interval; Auto-Correlation-like)
  pair   G = {k, g}, every g in k+1..J-NMIN, linear lam=1 (the nearest admissible lag + one far interval)
  mean   the in-window mean of all J errors (bias only);  zero  (no correction)
Estimators: each family with its config chosen on val (per lead); stacks = per window and variable, weights
w_c ~ exp(-(LOO_c / min LOO - 1) / tau) over the listed candidates + mean + zero, tau in TAUS (0 = argmin), LOO_c = the
candidate's LOO MSE on the variable's NE most recent demos (the same demo targets for all candidates):
  stack_adj = adj;  single_best = single;  stack_lag = adj + lin + rbf + single + pair (own-variable temporal intervals);
  stack_all = stack_lag + xvar.
Shuffle control (prefix sh_): the demo INPUTS are permuted, the outputs Y untouched. One random key per (window, variable,
position u) is shared by all candidates; each candidate reorders its own demo rows by the keys restricted to its demo range. So
every candidate keeps exactly its own demo outputs (its bias, and the LOO evaluation targets, identical to the real run and common
to all candidates) and only the input-output pairing is broken; mean and zero are unchanged. NSH independent key draws, SSE
averaged; evaluated at the SAME config as the real estimator. (Permuting outputs over all J positions would also move older
errors into a long-lag candidate's recent demo set and destroy its recent-level bias, overstating pairing.)
Score MSEred = 1 - sum |r - delta|^2 / sum |r|^2 pooled over windows and variables.
Pre-registered (ETTh2, A1, L = 1440, test, mean over leads 4-7): multi-lag in-context pairing is supported iff
  stack_lag - sh_stack_lag >= 0.02   AND   stack_lag - stack_adj >= 0.02.
Attribution (secondary): single_best, lin / rbf / pair families, stack_all - stack_lag (cross-variable). Weather (daily period =
3 patches), L = 720 and A0 are secondary. Windows overlap (stride 1), so the pooled numbers carry no naive error bars.
usage: python patchmem/icl_lag.py --data E|W [--L 720,1440] [--models A1] [--stride 1]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

P, NP, SEED = 48, 7, 3047
NMIN, NE = 6, 4
ADJ_M = (1, 2, 3)
LAMS = (0.3, 1.0, 3.0, 10.0)
SPANS = (4, 8, 12, 16, 22)
RSPANS, RLAMS, HS = (4, 8, 16), (0.3, 3.0), (0.5, 1.0, 2.0)
XSPANS, XLAMS = (1, 2, 4, 8), (1.0, 3.0, 10.0, 30.0)
TAUS = (0.0, 0.1, 0.3, 1.0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
_tt = torch.arange(P, dtype=torch.float64) - (P - 1) / 2
SL = _tt / (_tt ** 2).sum()


def feats(E):
    """E (..., P) -> (..., 2): patch mean and least-squares slope."""
    return torch.stack([E.mean(-1), (E * SL.to(E)).sum(-1)], -1)


def kridge(X, Y, Xq, lam, h=None):
    """Dual kernel ridge, unpenalised intercept. X (B, n, d), Y (B, n, P), Xq (B, q, d) -> pred (B, q, P), LOO residual (B, n, P)."""
    B, n, d = X.shape
    mu, sd = X.mean(1, keepdim=True), X.std(1, keepdim=True).clamp_min(1e-6)
    Z, Zq = (X - mu) / sd, (Xq - mu) / sd
    if h is None:
        K, Kq = Z @ Z.transpose(1, 2), Zq @ Z.transpose(1, 2)
    else:
        D, Dq = torch.cdist(Z, Z) ** 2, torch.cdist(Zq, Z) ** 2
        off = ~torch.eye(n, dtype=torch.bool, device=X.device)
        bw = (h * h) * D[:, off].median(-1).values.clamp_min(1e-6)[:, None, None]
        K, Kq = torch.exp(-D / bw), torch.exp(-Dq / bw)
    cm = K.mean(1, keepdim=True)                                                        # (B, 1, n) column means
    Kc = K - cm - cm.transpose(1, 2) + cm.mean(-1, keepdim=True)
    Kqc = Kq - Kq.mean(-1, keepdim=True) - cm + cm.mean(-1, keepdim=True)
    I = torch.eye(n, dtype=X.dtype, device=X.device)
    lam_ = lam * (torch.diagonal(Kc, dim1=1, dim2=2).mean(-1).clamp_min(1e-3))[:, None, None]
    Ainv = torch.linalg.inv(Kc + lam_ * I)
    yb = Y.mean(1, keepdim=True)
    al = Ainv @ (Y - yb)
    pred = yb + Kqc @ al
    fit = yb + Kc @ al
    Sd = 1.0 / n + 1 - lam_[:, :, 0] * torch.diagonal(Ainv, dim1=1, dim2=2)            # diag(Kc Ainv) = 1 - lam' diag(Ainv)
    return pred, (Y - fit) / (1 - Sd).clamp_min(1e-6)[..., None]


def build(Ech, Fp, k, G, xvar=False, keys=None):
    """Ech (B, J, P, C), Fp (B, J, C, 2); keys (B, C, J) shared random keys -> demo inputs permuted (null). Returns X (B, C, n, d),
    Y (B, C, n, P), Xq (B, C, d) or None."""
    B, J, _, C = Ech.shape
    gmax = max(G); n = J - gmax
    if n < NMIN:
        return None
    us = torch.arange(gmax, J, device=Ech.device)
    qs = [J - 1 + k - g for g in G]
    if xvar:
        X = torch.cat([Fp[:, us - g].flatten(2) for g in G], -1)[:, None].expand(B, C, n, -1)
        Xq = torch.cat([Fp[:, q].flatten(1) for q in qs], -1)[:, None].expand(B, C, -1)
    else:
        X = torch.cat([Fp[:, us - g].transpose(1, 2) for g in G], -1)                   # (B, C, n, 2|G|)
        Xq = torch.cat([Fp[:, q] for q in qs], -1)                                      # (B, C, 2|G|)
    Y = Ech[:, us].permute(0, 3, 1, 2)                                                  # (B, C, n, P)
    if keys is not None:
        perm = torch.argsort(keys[:, :, us], -1)                                        # (B, C, n)
        X = torch.gather(X, 2, perm[..., None].expand(-1, -1, -1, X.shape[-1]))
    return X, Y, Xq


def solve(X, Y, Xq, lam, h=None, pool=False):
    """-> pred (B, C, P), eval LOO MSE on the NE most recent demos of each variable (B, C)."""
    B, C, n, d = X.shape
    if pool:
        pr, res = kridge(X.reshape(B, C * n, d), Y.reshape(B, C * n, -1), Xq, lam, h)
        res = res.view(B, C, n, -1)
    else:
        pr, res = kridge(X.reshape(B * C, n, d), Y.reshape(B * C, n, -1), Xq.reshape(B * C, 1, d), lam, h)
        pr, res = pr.view(B, C, -1), res.view(B, C, n, -1)
    return pr, (res[:, :, -NE:] ** 2).mean((-1, -2))


def candidates(Ech, Fp, k, keys=None):
    """keys (B, C, J): shared random keys for the null (demo inputs permuted), None = real. Yields (family, cfg, pred (B, C, P), eloo (B, C))."""
    B, J, _, C = Ech.shape
    def run(G, lam, h=None, pool=False, xvar=False):
        b = build(Ech, Fp, k, G, xvar, keys)
        if b is None:
            return None
        return solve(*b, lam, h, pool)
    for m in ADJ_M:
        for lam in LAMS:
            o = run(tuple(range(k, k + m)), lam)
            if o: yield ('adj', f'm{m}_l{lam}') + o
    for S in SPANS:
        for lam in LAMS:
            for pool in (0, 1):
                o = run(tuple(range(k, k + S)), lam, pool=bool(pool))
                if o: yield ('lin', f'S{S}_l{lam}_p{pool}') + o
    for S in RSPANS:
        for lam in RLAMS:
            for h in HS:
                for pool in (0, 1):
                    o = run(tuple(range(k, k + S)), lam, h, bool(pool))
                    if o: yield ('rbf', f'S{S}_l{lam}_h{h}_p{pool}') + o
    for S in XSPANS:
        for lam in XLAMS:
            o = run(tuple(range(k, k + S)), lam, xvar=True)
            if o: yield ('xvar', f'S{S}_l{lam}') + o
    for g in range(k, J - NMIN + 1):
        o = run((g,), 1.0)
        if o: yield ('single', f'g{g}') + o
    for g in range(k + 1, J - NMIN + 1):
        o = run((k, g), 1.0)
        if o: yield ('pair', f'g{g}') + o
    Y = Ech.permute(0, 3, 1, 2)                                                         # (B, C, J, P) all errors
    yb = Y.mean(2)
    yield ('mean', 'all', yb, (((Y[:, :, -NE:] - yb[:, :, None]) * J / (J - 1)) ** 2).mean((-1, -2)))
    yield ('zero', '-', torch.zeros_like(yb), (Y[:, :, -NE:] ** 2).mean((-1, -2)))


def stack(cands, tau):
    pr = torch.stack([c[0] for c in cands], 0)                                          # (K, B, C, P)
    el = torch.stack([c[1] for c in cands], 0)                                          # (K, B, C)
    if tau == 0:
        w = torch.nn.functional.one_hot(el.argmin(0), len(cands)).permute(2, 0, 1).to(pr.dtype)
    else:
        w = torch.softmax(-(el / el.min(0).values.clamp_min(1e-12) - 1) / tau, 0)
    return (w[..., None] * pr).sum(0)


def run_split(F, Hs, gen, chunk, nsh):
    """F (N, H, C) future residuals, Hs (N, J, P, C) history (index 0 = most recent). Returns {name: {cfg: [sse(NP), ss0(NP), valid(NP)]}}."""
    out = {}
    def acc(name, cfg, k, d, r, w=1.0):
        o = out.setdefault(name, {}).setdefault(cfg, [np.zeros(NP), np.zeros(NP), np.zeros(NP, dtype=bool)])
        o[0][k - 1] += w * ((r - d) ** 2).sum().item(); o[1][k - 1] += w * (r ** 2).sum().item(); o[2][k - 1] = True
    N = len(F)
    for i0 in range(0, N, chunk):
        Ft = torch.as_tensor(F[i0:i0 + chunk], dtype=torch.float64, device=dev)
        Ech = torch.as_tensor(np.ascontiguousarray(Hs[i0:i0 + chunk, ::-1]), dtype=torch.float64, device=dev)   # chronological
        Fp = feats(Ech.permute(0, 1, 3, 2))                                             # (B, J, C, 2)
        B, J, _, C = Ech.shape
        Ks = [('', None, 1.0)] + [('sh_', torch.rand(B, C, J, generator=gen, device=dev, dtype=torch.float64), 1.0 / nsh) for _ in range(nsh)]
        for k in range(1, NP + 1):
            r = Ft[:, (k - 1) * P:k * P].transpose(1, 2)                                # (B, C, P)
            for pre, keys, w in Ks:
                fam = {}
                for f, cfg, pr, el in candidates(Ech, Fp, k, keys):
                    acc(pre + f, cfg, k, pr, r, w)
                    fam.setdefault(f, []).append((pr, el))
                base = fam['mean'] + fam['zero']
                for nm, fs in STACKS:
                    cs = [c for f in fs for c in fam.get(f, [])]
                    if not cs:
                        continue
                    for tau in TAUS:
                        acc(pre + nm, f't{tau}', k, stack(cs + base, tau), r, w)
    return out


STACKS = (('stack_adj', ('adj',)), ('single_best', ('single',)), ('stack_lag', ('adj', 'lin', 'rbf', 'single', 'pair')),
          ('stack_all', ('adj', 'lin', 'rbf', 'single', 'pair', 'xvar')))
FAMS = ('adj', 'lin', 'rbf', 'pair', 'xvar', 'single', 'mean', 'stack_adj', 'single_best', 'stack_lag', 'stack_all')


def report(ent, tag):
    """Config per family and lead chosen on val; prints test (and the shuffle control at the same config)."""
    def ms(d):
        return d[0] / np.maximum(d[1], 1e-12)
    sel, rows = {}, {}
    for nm in FAMS:
        if nm not in ent['val'] or nm in ('single', 'pair'):
            continue
        sel[nm] = []
        vt, vs = np.full(NP, np.nan), np.full(NP, np.nan)
        for k in range(NP):
            ok = {c: 1 - ms(d)[k] for c, d in ent['val'][nm].items() if d[2][k]}
            if not ok:
                sel[nm].append(None); continue
            c = max(ok, key=ok.get); sel[nm].append(c)
            vt[k] = 1 - ms(ent['test'][nm][c])[k]
            if 'sh_' + nm in ent['test']:
                vs[k] = 1 - ms(ent['test']['sh_' + nm][c])[k]
        rows[nm] = (vt, vs)
    print(f'===== {tag} | test MSEred by lead 1..7 (config per lead chosen on val) | [shuffle control at the same config]', flush=True)
    f = lambda v: ' '.join('   -  ' if np.isnan(x) else f'{x:+.3f}' for x in v)
    for nm, (vt, vs) in rows.items():
        print(f'  {nm:11s} {f(vt)} | sh {f(vs)} | L4-7 {np.nanmean(vt[3:]):+.3f} sh {np.nanmean(vs[3:]):+.3f}', flush=True)
    print('  selected: ' + '; '.join(f'{nm} {sel[nm]}' for nm in ('lin', 'rbf', 'xvar', 'stack_lag', 'stack_all') if nm in sel), flush=True)
    return sel, {nm: dict(test=v[0].tolist(), sh=v[1].tolist()) for nm, v in rows.items()}


def main():
    from patchmem.diag_rollout import official_args
    from patchmem.plug_a0 import load_a0, cache_split
    from patchmem.resid_cov import CFG, H, split_x, a1_resid, a0_resid
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W'])
    ap.add_argument('--L', default='720,1440'); ap.add_argument('--models', default='A1'); ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--chunk', type=int, default=128); ap.add_argument('--nsh', type=int, default=3); ap.add_argument('--tag', default='v3')
    a_ = ap.parse_args()
    cf = CFG[a_.data]; res = {}; mdls = a_.models.split(',')
    os.environ.pop('R1_OBS', None)
    from patchmem.exp_r1 import Exp_R1
    ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {cf["r1des"]} --seed {SEED}').split())
    ex = Exp_R1(ar)
    ex.model.load_state_dict(torch.load(f'checkpoints/{cf["r1ck"]}/checkpoint.pth', map_location=dev))
    gen = torch.Generator(device=dev).manual_seed(0)
    for L in [int(v) for v in a_.L.split(',')]:
        t0 = time.time(); J = L // P - 1
        R = {m: {} for m in mdls}
        if 'A0' in mdls:
            aa = official_args((cf['common'] + ' ' + cf['a0'] + f' --model_id {cf["a0id"].format(L=L)} --seq_len {L} --des ExpA0s{SEED} --seed {SEED}').split())
            a0, _, order = load_a0(aa, cf['a0ck'].format(L=L), SEED)
        for sp in ('val', 'test'):
            X = split_x(ar, L, sp)
            origins = list(range(L + J * P, len(X) - H + 1))[::a_.stride]
            if 'A1' in mdls:
                R['A1'][sp] = a1_resid(ex, X, L, origins, dev, mmax=J)
            if 'A0' in mdls:
                Y = cache_split(a0, dev, X, L, H, order)[0].numpy()
                R['A0'][sp] = a0_resid(Y, X, L, origins, mmax=J)
        if 'A0' in mdls:
            del a0; torch.cuda.empty_cache()
        for mdl in mdls:
            ent = {sp: run_split(*R[mdl][sp], gen, a_.chunk, a_.nsh) for sp in ('val', 'test')}
            sel, rows = report(ent, f'{a_.data} {mdl} L={L} (J={J}) {len(R[mdl]["test"][0])} test windows | {time.time() - t0:.0f}s')
            res[f'{mdl}_L{L}'] = dict(selected=sel, rows=rows,
                                      raw={sp: {nm: {c: [v[0].tolist(), v[1].tolist(), v[2].tolist()] for c, v in d.items()} for nm, d in e.items()} for sp, e in ent.items()})
            json.dump(res, open(f'patchmem/logs/server/icl_lag_{a_.tag}_{a_.data}_s{SEED}.json', 'w'))
    print('DONE')


if __name__ == '__main__':
    main()
