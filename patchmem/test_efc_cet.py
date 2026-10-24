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
    test_ced_indexing(); test_no_lookahead(); test_recovers_transport(); print('ok')
