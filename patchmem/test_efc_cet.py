"""CPU tests for patchmem/efc_cet.py: CED indexing, no use of unmatured errors, recovery of a linear error transport."""
import torch
import patchmem.efc_cet as ec


def test_ced_indexing():
    N, H, C = 10, 3, 2
    E = torch.arange(N * 7 * C, dtype=torch.float32).reshape(N, 7, C)
    D, valid = ec.ced(E, H)
    for t in range(H, N):
        for h in range(1, H + 1):
            assert torch.equal(D[t, h - 1], E[t - h, h - 1])
    assert not bool(valid[:H].any()) and bool(valid[H:].all())


def test_no_lookahead():
    """At origin t only y[..t-1] is observed: errors E[i, h-1] with target index i+h-1 >= t (not yet observed) must not change the
    correction at t."""
    g = torch.Generator().manual_seed(0)
    N, H, C = 60, 3, 2
    E = torch.randn(N, 7, C, generator=g)
    c1, a1 = ec.stream(E.clone(), H, 0, H)
    t = 40
    E2 = E.clone()
    for i in range(N):
        for h in range(1, 8):
            if i + h - 1 >= t:                                                           # unobserved target at time t
                E2[i, h - 1] += 5.0
    c2, a2 = ec.stream(E2, H, 0, H)
    assert torch.allclose(c1[t], c2[t], atol=1e-5) and bool(a1[t])


def test_recovers_transport():
    """E_t = D_t A (+ small noise): with alpha calibrated, the correction removes most of the error."""
    g = torch.Generator().manual_seed(1)
    N, H, C = 400, 3, 2
    A = 0.6 * torch.randn(H, H, generator=g)
    E = torch.zeros(N, 7, C)
    for t in range(N):                                                                   # build sequentially: D_t from earlier origins
        D = torch.stack([E[t - h, h - 1] if t - h >= 0 else torch.zeros(C) for h in range(1, H + 1)])   # (H, C)
        E[t, :H] = (D.T @ A).T + torch.randn(H, C, generator=g)
    c, act = ec.stream(E, H, 0, H)
    late = slice(200, N)
    before = E[late, :H].square().mean(); after = (E[late, :H] - c[late]).square().mean()
    assert after < before * 0.9 and bool(act[late].all()), (before, after)


def test_row_feature_indexing_no_lookahead_and_transport():
    N, H, C = 10, 3, 2
    E = torch.arange(N * 7 * C, dtype=torch.float32).reshape(N, 7, C)
    F, valid = ec.row_feat(E, H)
    assert all(torch.equal(F[t], E[t - H, :H]) for t in range(H, N)) and not bool(valid[:H].any()) and bool(valid[H:].all())
    g = torch.Generator().manual_seed(0)
    N = 60
    E = torch.randn(N, 7, C, generator=g)
    c1, a1 = ec.stream(E.clone(), H, 0, H, feat='row')
    t = 40
    E2 = E.clone()
    for i in range(N):
        for h in range(1, 8):
            if i + h - 1 >= t:
                E2[i, h - 1] += 5.0
    c2, _ = ec.stream(E2, H, 0, H, feat='row')
    assert torch.allclose(c1[t], c2[t], atol=1e-5) and bool(a1[t])
    g = torch.Generator().manual_seed(1)
    N = 400
    A = 0.6 * torch.randn(H, H, generator=g)
    E = torch.zeros(N, 7, C)
    for t in range(N):
        prev = E[t - H, :H] if t >= H else torch.zeros(H, C)
        E[t, :H] = (prev.T @ A).T + torch.randn(H, C, generator=g)
    c, act = ec.stream(E, H, 0, H, feat='row')
    late = slice(200, N)
    assert (E[late, :H] - c[late]).square().mean() < 0.9 * E[late, :H].square().mean() and bool(act[late].all())


if __name__ == '__main__':
    test_row_feature_indexing_no_lookahead_and_transport()
    test_win_matches_brute_force_and_window()
    test_ced_indexing(); test_no_lookahead(); test_recovers_transport(); print('ok')


def _brute_win(E, H, nv, feat, L, calibrate=True):
    """Direct recomputation of the lookback-limited stream from the explicit origin set at each t."""
    N, _, C = E.shape
    D, valid = (ec.ced if feat == 'ced' else ec.row_feat)(E, H)
    rD = D.permute(0, 2, 1).double(); rE = E[:, :H].permute(0, 2, 1).double()
    eye = ec.LAM * torch.eye(H, dtype=torch.float64)
    out = torch.zeros(N, H, C); act = torch.zeros(N, dtype=torch.bool)
    for t in range(N):
        M = [i for i in range(max(0, t - L + H), t - H + 1) if bool(valid[i])]
        if len(M) < nv + 1 or not bool(valid[t]):
            continue
        fit = lambda idx: torch.linalg.solve(eye + sum(rD[i].T @ rD[i] for i in idx), sum(rD[i].T @ rE[i] for i in idx))
        al = 1.0
        if calibrate:
            Wv = fit(M[:-nv]); V = M[-nv:]
            Q = torch.stack([rD[i] @ Wv for i in V]); Y = torch.stack([rE[i] for i in V])
            al = float(((Q * Y).sum() / (Q.square().sum() + 1e-8)).clamp(0, 1))
        out[t] = (al * (rD[t] @ fit(M))).float().T; act[t] = True
    return out, act


def test_win_matches_brute_force_and_window():
    g = torch.Generator().manual_seed(2)
    N, H, C, L = 70, 3, 2, 16
    E = torch.randn(N, 7, C, generator=g)
    for feat in ('ced', 'row'):
        for cal in (True, False):
            c, a = ec.stream(E, H, 0, 2, cal, feat, win=L)
            cb, ab = _brute_win(E, H, 2, feat, L, cal)
            assert torch.equal(a, ab) and torch.allclose(c, cb, atol=1e-5), (feat, cal, (c - cb).abs().max())
        t = 50                                                                            # targets outside [t-L, t) must not matter
        E2 = E.clone()
        for i in range(N):
            for h in range(1, 8):
                if i + h - 1 >= t or i + h - 1 < t - L:
                    E2[i, h - 1] += 5.0
        for cal in (True, False):
            c, a = ec.stream(E, H, 0, 2, cal, feat, win=L)
            c2, _ = ec.stream(E2, H, 0, 2, cal, feat, win=L)
            assert torch.allclose(c[t], c2[t], atol=1e-5) and bool(a[t]), (feat, cal)
        assert not torch.equal(ec.stream(E, H, 0, 2, False, feat)[0][t], ec.stream(E2, H, 0, 2, False, feat)[0][t])   # full stream does see it
