import torch

import patchmem.eval_fc_vs_b as fb


def moments(Z, e):
    return Z.T @ Z, Z.T @ e, e @ e


def test_solve_and_sse_match_lstsq():
    g = torch.Generator().manual_seed(0)
    Z = torch.randn(300, 3, generator=g, dtype=torch.float64)
    e = Z @ torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64) + torch.randn(300, generator=g, dtype=torch.float64)
    G, b, ee = moments(Z, e)
    for idx in ((0,), (0, 1), (0, 2), (0, 1, 2)):
        w = fb.solve(G, b, idx)
        ref = torch.linalg.lstsq(Z[:, list(idx)], e[:, None]).solution[:, 0]
        assert torch.allclose(w[list(idx)], ref, atol=1e-6)
        assert all(float(w[i]) == 0 for i in range(3) if i not in idx)
        assert torch.allclose(fb.sse(G, b, ee, w), (e - Z @ w).square().sum(), rtol=1e-9)


def test_single_is_fc_weight():
    g = torch.Generator().manual_seed(1)
    Z = torch.randn(200, 3, generator=g, dtype=torch.float64)
    e = 0.4 * (Z[:, 0] + Z[:, 1]) + 0.1 * torch.randn(200, generator=g, dtype=torch.float64)
    G, b, _ = moments(Z, e)
    w = fb.single(G, b, (0, 1))
    d = Z[:, 0] + Z[:, 1]
    want = float((e * d).sum() / d.square().sum())
    assert abs(float(w[0]) - want) < 1e-12 and float(w[0]) == float(w[1]) and float(w[2]) == 0
    e2 = -e
    G2, b2, _ = moments(Z, e2)
    assert float(fb.single(G2, b2, (0, 1)).abs().sum()) == 0.0
