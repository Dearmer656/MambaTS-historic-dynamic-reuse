"""Unit tests for patchmem/icl_lag.py: dual solver = primal ridge, exact LOO, RBF LOO, lag indexing, shuffle keeps rows."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import patchmem.icl_lag as M
torch.manual_seed(0); dt = torch.float64
B, n, d, P = 3, 9, 4, 5
X, Y, Xq = torch.randn(B, n, d, dtype=dt), torch.randn(B, n, P, dtype=dt), torch.randn(B, 2, d, dtype=dt)
for h in (None, 1.0):
    pr, res = M.kridge(X, Y, Xq, 1.0, h)
    for i in range(n):                                                      # brute-force LOO with the SAME standardisation/kernel bandwidth is not
        pass                                                                # identical (they are re-estimated); check the linear smoother identity instead
# linear: dual == primal ridge on standardised features with unpenalised intercept
mu, sd = X.mean(1, keepdim=True), X.std(1, keepdim=True)
Z, Zq = (X - mu) / sd, (Xq - mu) / sd
pr, res = M.kridge(X, Y, Xq, 1.0)
Zc = Z - Z.mean(1, keepdim=True)
lam = 1.0 * (Zc @ Zc.transpose(1, 2)).diagonal(dim1=1, dim2=2).mean(-1)
W = torch.linalg.solve(Zc.transpose(1, 2) @ Zc + lam[:, None, None] * torch.eye(d, dtype=dt), Zc.transpose(1, 2) @ (Y - Y.mean(1, keepdim=True)))
pp = Y.mean(1, keepdim=True) + (Zq - Z.mean(1, keepdim=True)) @ W
assert torch.allclose(pr, pp, atol=1e-9), (pr - pp).abs().max()
# exact LOO for the fixed smoother (fixed standardisation and lam'): refit without row i
for b in range(B):
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        Zk, Yk = Z[b, keep], Y[b, keep]
        zm, ym = Zk.mean(0), Yk.mean(0)
        Wk = torch.linalg.solve((Zk - zm).T @ (Zk - zm) + lam[b] * torch.eye(d, dtype=dt), (Zk - zm).T @ (Yk - ym))
        e = Y[b, i] - (ym + (Z[b, i] - zm) @ Wk)
        assert torch.allclose(res[b, i], e, atol=1e-8), (b, i, (res[b, i] - e).abs().max())
# RBF: LOO identity for the fixed kernel matrix (centred kernel ridge == kernel ridge with unpenalised intercept)
pr, res = M.kridge(X, Y, Xq, 0.5, 1.0)
D = torch.cdist(Z, Z) ** 2; off = ~torch.eye(n, dtype=torch.bool)
bw = D[:, off].median(-1).values[:, None, None]; K = torch.exp(-D / bw)
for b in range(1):
    lamb = None
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        # intercept-unpenalised kernel ridge: solve [[K+l'I, 1],[1^T, 0]] system on the kept rows with the full-data lam'
        Kc = K[b] - K[b].mean(0, keepdim=True) - K[b].mean(1, keepdim=True) + K[b].mean()
        lp = 0.5 * Kc.diagonal().mean()
        Kk = K[b][keep][:, keep]; m = n - 1
        A = torch.zeros(m + 1, m + 1, dtype=dt); A[:m, :m] = Kk + lp * torch.eye(m, dtype=dt); A[:m, m] = 1; A[m, :m] = 1
        rhs = torch.zeros(m + 1, P, dtype=dt); rhs[:m] = Y[b, keep]
        s = torch.linalg.solve(A, rhs)
        e = Y[b, i] - (K[b][i, keep] @ s[:m] + s[m])
        assert torch.allclose(res[b, i], e, atol=1e-7), (i, (res[b, i] - e).abs().max())
# full-data RBF prediction matches the bordered system
A = torch.zeros(n + 1, n + 1, dtype=dt); A[:n, :n] = K[0] + lp * torch.eye(n, dtype=dt); A[:n, n] = 1; A[n, :n] = 1
rhs = torch.zeros(n + 1, P, dtype=dt); rhs[:n] = Y[0]; s = torch.linalg.solve(A, rhs)
Kq = torch.exp(-torch.cdist(Zq[0], Z[0]) ** 2 / bw[0])
assert torch.allclose(pr[0], Kq @ s[:n] + s[n], atol=1e-8)
# lag indexing: errors periodic with period g0 -> the single-lag-g0 candidate is near perfect, other lags are not
J, C, g0 = 29, 2, 5
cs = torch.randn(g0, 2, C, dtype=dt)
tt = (torch.arange(M.P, dtype=dt) - (M.P - 1) / 2)
base = cs[:, 0, None] + cs[:, 1, None] * tt[None, :, None] * 0.05            # (g0, P, C): linear in (mean, slope) -> exactly learnable
E = torch.stack([base[t % g0] for t in range(J + M.NP)], 0)[None]                  # (1, J+7, P, C)
Ech, fut = E[:, :J], E[:, J:]
Fp = M.feats(Ech.permute(0, 1, 3, 2))
for k in (1, 3, 5):
    for g in (g0, g0 + 1) if k <= g0 else (2 * g0, 2 * g0 + 1):
        X_, Y_, Xq_ = M.build(Ech, Fp, k, (g,))
        # demo inputs must be phi of e_{u-g}: check row 0 (u = g)
        assert torch.allclose(X_[0, :, 0], Fp[0, 0], atol=0), 'demo index'
        assert torch.allclose(Xq_[0], Fp[0, J - 1 + k - g]), 'query index'
        pr, _ = M.solve(X_, Y_, Xq_, 1e-8)
        err = ((pr[0] - fut[0, k - 1].T) ** 2).mean().item()
        print(f'k={k} g={g} mse={err:.2e}')
        if g % g0 == 0:
            assert err < 1e-3
# xvar and adjacent shapes, shuffle keeps the multiset of rows
X_, Y_, Xq_ = M.build(Ech, Fp, 2, (2, 3, 4), xvar=True); assert X_.shape[-1] == 2 * C * 3 and Xq_.shape == (1, C, 2 * C * 3)
g = torch.Generator().manual_seed(0); keys = torch.rand(1, C, J, generator=g, dtype=dt)
# the null permutes demo INPUTS only: outputs identical to the real run, input rows a permutation of the real rows
X0, Y0, Q0 = M.build(Ech, Fp, 1, (1, 2, 3)); Xs, Ys, Qs = M.build(Ech, Fp, 1, (1, 2, 3), keys=keys)
assert torch.equal(Y0, Ys) and torch.equal(Q0, Qs) and not torch.equal(X0, Xs)
assert torch.allclose(Xs.sort(2).values, X0.sort(2).values)
# shared keys: a nested demo range gets the restriction of the same random order
n0 = X0.shape[2]; us = torch.arange(J - n0, J); perm = torch.argsort(keys[0, 0, us])
assert torch.equal(Xs[0, 0], X0[0, 0, perm])
# pooled solve runs and returns per-variable predictions
pr, el = M.solve(X_, Y_, Xq_, 1.0, pool=True); assert pr.shape == (1, C, M.P) and el.shape == (1, C)
# candidate generator end to end
M.dev = 'cpu'
gen = torch.Generator().manual_seed(0)
fams = {c[0] for c in M.candidates(Ech, Fp, 3, keys)}
assert fams == {'adj', 'lin', 'rbf', 'xvar', 'single', 'pair', 'mean', 'zero'}, fams
import numpy as np
Ffut = fut[0].reshape(1, -1, C).repeat(4, 1, 1).numpy(); Hs = Ech.flip(1).repeat(4, 1, 1, 1).numpy()
out = M.run_split(Ffut, Hs, gen, 2, 2)
assert {'stack_lag', 'sh_stack_lag', 'stack_all', 'pair'} <= set(out)
print('stack_lag t0 MSEred', 1 - out['stack_lag']['t0.0'][0] / out['stack_lag']['t0.0'][1], 'sh', 1 - out['sh_stack_lag']['t0.0'][0] / out['sh_stack_lag']['t0.0'][1])
print('ALL PASS')
