"""PAT-307 offline in-context-learning probe on the residuals (no training of any model; A1 / A0 seed-3047 checkpoints, frozen).
Question: does the lookback contain demonstration pairs from which a correction rule can be inferred AT INFERENCE (and that a rule
fitted on train cannot supply)? Residuals as patchmem/resid_cov.py (dataset StandardScaler units): e_u = 1-patch-ahead error of
lookback patch u (A1: in-window teacher-forced; A0: first patch of the forecast issued at the past origin), u in chronological
order 0..J-1 (J = L/P - 1, u = J-1 the most recent); target r_k = the rollout residual of future lead k = 1..7.
Lead-aligned demonstrations (per window, variable c, lead k, history length m): input = phi(e_{u-k-m+1}, ..., e_{u-k}), output = e_u,
for every u with u - k - m + 1 >= 0; query input = phi(e_{J-m}, ..., e_{J-1}) (the m most recent patches), predicting r_k.
phi(patch) = (mean, least-squares slope over the 48 steps) -> 2m features, plus an intercept (not penalised).
Estimators (delta added to the forecast; MSEred = 1 - sum |r - delta|^2 / sum |r|^2, pooled over windows and variables):
  icl_ridge   W = argmin sum_demo |e_u - [phi, 1] W|^2 + lam |W_phi|^2 (closed form, per window), delta = [phi_q, 1] W
  icl_null    icl_ridge, but delta = 0 when the window's leave-one-out SSE of the ridge >= the SSE of predicting 0 (abstain)
  icl_nw      Nadaraya-Watson: delta = sum_u softmax_u(-|phi_q - phi_u|^2 / (2 h^2)) e_u, phi standardised inside the window
              (zeroth order, the O-like in-context kernel regression)
  icl_mean    delta = the mean of the demonstration outputs (in-context bias only)
  train_lin   the same features, ONE linear map per (variable, lead) fitted on all train windows with the true lead-k targets
              (not in-context; the fixed-rule baseline)
  shuffle     icl_ridge with the demonstration outputs permuted inside each window (pairing broken; control)
Hyper-parameters (m in {1,2,3}; lam for icl_* AND train_lin; h) are chosen on val by the mean MSEred over the leads that have
demos and reported on test (for L = 384/720 large m has no demos at far leads, so configs are chosen on different lead sets there).
Feature order: most recent patch first, identically for demos and query. LOO uses the window's fixed lambda scale (exact for that
smoother); a 1e-9 jitter on the diagonal (incl. the intercept) only for numerical stability.
Pre-registered: ICL is supported if on ETTh2 A1 L=1440 test, leads 3-7, max(icl_ridge, icl_null) - train_lin >= 0.03 AND shuffle <= 0.
usage: python patchmem/icl_probe.py --data E|W"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from patchmem.diag_rollout import official_args
from patchmem.plug_a0 import load_a0, cache_split
from patchmem.resid_cov import CFG, P, H, NP, SEED, split_x, a1_resid, a0_resid

MS = (1, 2, 3)
LAMS = (0.1, 1.0, 10.0)          # relative to the mean diagonal of the demo Gram
HS = (0.5, 1.0, 2.0)             # NW bandwidth in standardised units
dev = 'cuda'
_tt = torch.arange(P, dtype=torch.float64) - (P - 1) / 2
SL = (_tt / (_tt ** 2).sum())                                                           # slope weights


def feats(E):
    """E (..., P) -> (..., 2): patch mean and least-squares slope."""
    return torch.stack([E.mean(-1), (E * SL.to(E.device)).sum(-1)], -1)


def demos(Ech, k, m):
    """Ech (B, J, P) chronological errors of one variable. Returns X (B, n, 2m), Y (B, n, P), Xq (B, 2m); n = J - k - m + 1 demos."""
    B, J, _ = Ech.shape
    n = J - k - m + 1
    if n < 2:
        return None
    Fp = feats(Ech)                                                                     # (B, J, 2)
    us = torch.arange(k + m - 1, J, device=Ech.device)                                  # demo targets
    X = torch.cat([Fp[:, us - k - i] for i in range(m)], -1)                            # (B, n, 2m): lag k, k+1, ..
    Y = Ech[:, us]
    Xq = torch.cat([Fp[:, J - 1 - i] for i in range(m)], -1)                            # most recent first, same order
    return X, Y, Xq


def ridge(X, Y, Xq, lam, loo=False):
    """Per-window ridge with an unpenalised intercept. X (B, n, d), Y (B, n, P), Xq (B, d). Returns pred (B, P) [, loo_sse, null_sse]."""
    B, n, d = X.shape
    Xa = torch.cat([X, torch.ones(B, n, 1, dtype=X.dtype, device=X.device)], -1)
    G = Xa.transpose(1, 2) @ Xa
    sc = torch.diagonal(G[:, :d, :d], dim1=1, dim2=2).mean(-1).clamp_min(1e-12)       # (B,)
    R = torch.zeros(d + 1, d + 1, dtype=X.dtype, device=X.device); R[:d, :d] = torch.eye(d)
    A = G + lam * sc[:, None, None] * R + 1e-9 * torch.eye(d + 1, dtype=X.dtype, device=X.device)
    Ainv = torch.linalg.inv(A)
    W = Ainv @ Xa.transpose(1, 2) @ Y                                                   # (B, d+1, P)
    pred = torch.cat([Xq, torch.ones(B, 1, dtype=X.dtype, device=X.device)], -1)[:, None] @ W
    if not loo:
        return pred[:, 0]
    Hd = torch.einsum('bnd,bde,bne->bn', Xa, Ainv, Xa)                                  # hat diagonal
    res = (Y - Xa @ W) / (1 - Hd).clamp_min(1e-6)[..., None]
    return pred[:, 0], (res ** 2).sum((1, 2)), (Y ** 2).sum((1, 2))


def nw(X, Y, Xq, h):
    mu, sd = X.mean(1, keepdim=True), X.std(1, keepdim=True).clamp_min(1e-6)
    Z, Zq = (X - mu) / sd, (Xq[:, None] - mu) / sd
    w = torch.softmax(-((Z - Zq) ** 2).sum(-1) / (2 * h * h), -1)                      # (B, n)
    return (w[..., None] * Y).sum(1)


def run_split(F, Hs, est, cfgs, train_maps=None, gen=None):
    """F (N, H, C) future residuals, Hs (N, J, P, C) history (j = 1 most recent). Returns {est: {cfg: (sse (NP,), ss0 (NP,))}}."""
    Ft = torch.as_tensor(F, dtype=torch.float64, device=dev)
    Ech_all = torch.as_tensor(np.ascontiguousarray(Hs[:, ::-1]), dtype=torch.float64, device=dev)   # chronological
    N, J, _, C = Ech_all.shape
    out = {}
    for c in range(C):
        Ech = Ech_all[..., c]
        for k in range(1, NP + 1):
            r = Ft[:, (k - 1) * P:k * P, c]
            s0 = (r ** 2).sum().item()
            for m in MS:
                dm = demos(Ech, k, m)
                if dm is None:
                    continue
                X, Y, Xq = dm
                for name, cfg, fn in cfgs(X, Y, Xq, m, k, c, train_maps, gen):
                    if name not in est:
                        continue
                    d = fn()
                    o = out.setdefault(name, {}).setdefault(cfg, [np.zeros(NP), np.zeros(NP), np.zeros(NP, dtype=bool)])
                    o[0][k - 1] += ((r - d) ** 2).sum().item(); o[1][k - 1] += s0; o[2][k - 1] = True
    return out


def cfgs(X, Y, Xq, m, k, c, train_maps, gen):
    for lam in LAMS:
        yield 'icl_ridge', f'm{m}_l{lam}', lambda lam=lam: ridge(X, Y, Xq, lam)
        def null(lam=lam):
            p, l1, l0 = ridge(X, Y, Xq, lam, loo=True)
            return p * (l1 < l0).to(p.dtype)[:, None]
        yield 'icl_null', f'm{m}_l{lam}', null
        def shuf(lam=lam):
            perm = torch.argsort(torch.rand(Y.shape[0], Y.shape[1], generator=gen, device='cpu'), 1).to(Y.device)
            return ridge(X, torch.gather(Y, 1, perm[..., None].expand_as(Y)), Xq, lam)
        yield 'shuffle', f'm{m}_l{lam}', shuf
    for h in HS:
        yield 'icl_nw', f'm{m}_h{h}', lambda h=h: nw(X, Y, Xq, h)
    yield 'icl_mean', f'm{m}', lambda: Y.mean(1)
    if train_maps is not None:
        for lam in LAMS:
            Wt = train_maps[(m, k, c, lam)]
            yield 'train_lin', f'm{m}_l{lam}', lambda Wt=Wt: torch.cat([Xq, torch.ones(Xq.shape[0], 1, dtype=Xq.dtype, device=Xq.device)], -1) @ Wt


def fit_train_maps(F, Hs):
    """train_lin: per (m, lead k, variable c), ridge of the TRUE lead-k residual (P) on the query features [phi_q, 1] over all train windows."""
    Ft = torch.as_tensor(F, dtype=torch.float64, device=dev)
    Ech_all = torch.as_tensor(np.ascontiguousarray(Hs[:, ::-1]), dtype=torch.float64, device=dev)
    N, J, _, C = Ech_all.shape
    maps = {}
    for c in range(C):
        Fp = feats(Ech_all[..., c])
        for m in MS:
            Xq = torch.cat([Fp[:, J - 1 - i] for i in range(m)], -1)
            Xa = torch.cat([Xq, torch.ones(N, 1, dtype=Xq.dtype, device=dev)], -1)
            G = Xa.T @ Xa; d = Xq.shape[1]
            for lam in LAMS:
                R = torch.zeros(d + 1, d + 1, dtype=G.dtype, device=dev); R[:d, :d] = torch.eye(d, dtype=G.dtype, device=dev) * lam * torch.diagonal(G)[:d].mean() / N
                for k in range(1, NP + 1):
                    maps[(m, k, c, lam)] = torch.linalg.solve(G + R, Xa.T @ Ft[:, (k - 1) * P:k * P, c])
    return maps


def finish(o):
    return {nm: {cf: dict(msered=(1 - v[0] / v[1]).tolist(), valid=v[2].tolist()) for cf, v in d.items()} for nm, d in o.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True, choices=['E', 'W']); ap.add_argument('--L', default='384,720,1440')
    a_ = ap.parse_args()
    cf = CFG[a_.data]; res = {}
    os.environ.pop('R1_OBS', None)
    from patchmem.exp_r1 import Exp_R1
    ar = official_args((cf['common'] + ' ' + cf['r1'] + f' --des {cf["r1des"]} --seed {SEED}').split())
    ex = Exp_R1(ar)
    ex.model.load_state_dict(torch.load(f'checkpoints/{cf["r1ck"]}/checkpoint.pth', map_location=dev))
    gen = torch.Generator().manual_seed(0)
    est = ('icl_ridge', 'icl_null', 'icl_nw', 'icl_mean', 'train_lin', 'shuffle')
    for L in [int(v) for v in a_.L.split(',')]:
        t0 = time.time(); J = L // P - 1
        aa = official_args((cf['common'] + ' ' + cf['a0'] + f' --model_id {cf["a0id"].format(L=L)} --seq_len {L} --des ExpA0s{SEED} --seed {SEED}').split())
        a0, _, order = load_a0(aa, cf['a0ck'].format(L=L), SEED)
        R = {'A1': {}, 'A0': {}}
        for sp in ('train', 'val', 'test'):
            X = split_x(ar, L, sp)
            origins = list(range(L + J * P, len(X) - H + 1))
            R['A1'][sp] = a1_resid(ex, X, L, origins, dev, mmax=J)
            Y = cache_split(a0, dev, X, L, H, order)[0].numpy()
            R['A0'][sp] = a0_resid(Y, X, L, origins, mmax=J)
        del a0; torch.cuda.empty_cache()
        for mdl in ('A1', 'A0'):
            maps = fit_train_maps(*R[mdl]['train'])
            ent = {}
            for sp in ('val', 'test'):
                ent[sp] = finish(run_split(*R[mdl][sp], est, cfgs, maps, gen))
            sel = {}
            for nm in est:                                                              # choose the config on val (mean over valid leads)
                sc = {c_: np.mean([v for v, ok in zip(d['msered'], d['valid']) if ok]) for c_, d in ent['val'][nm].items()}
                sel[nm] = max(sc, key=sc.get)
            ent['selected'] = sel
            res[f'{mdl}_L{L}'] = ent
            print(f'===== {a_.data} {mdl} L={L} (J={J}) | test MSEred by lead 1..7 (config chosen on val) | {time.time() - t0:.0f}s', flush=True)
            for nm in est:
                v = ent['test'][nm][sel[nm]]
                vv = ent['val'][nm][sel[nm]]
                print(f'  {nm:10s} {sel[nm]:9s} test ' + ' '.join(f'{x:+.3f}' if ok else '   -  ' for x, ok in zip(v['msered'], v['valid'])) +
                      ' | val ' + ' '.join(f'{x:+.3f}' if ok else '   -  ' for x, ok in zip(vv['msered'], vv['valid'])), flush=True)
        json.dump(res, open(f'patchmem/logs/server/icl_probe_{a_.data}_s{SEED}.json', 'w'))
    print('DONE')


if __name__ == '__main__':
    main()
