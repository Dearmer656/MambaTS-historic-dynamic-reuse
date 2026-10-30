import torch

import patchmem.eval_fc_stack as fs


def test_box_ls_matches_grid():
    """box-constrained 2-var LS == brute force over a fine grid of [0, 1]^2."""
    g = torch.Generator().manual_seed(0)
    grid = torch.linspace(0, 1, 401, dtype=torch.float64)
    w1, w2 = torch.meshgrid(grid, grid, indexing='ij')
    for _ in range(200):
        X = torch.randn(50, 2, generator=g, dtype=torch.float64) * torch.rand(2, generator=g, dtype=torch.float64)
        y = X @ (torch.rand(2, generator=g, dtype=torch.float64) * 3 - 1) + torch.randn(50, generator=g, dtype=torch.float64)
        A, b = X.T @ X, X.T @ y
        a1, a2 = fs.box_ls(float(A[0, 0]), float(A[0, 1]), float(A[1, 1]), float(b[0]), float(b[1]))
        q = lambda u, v: A[0, 0] * u ** 2 + 2 * A[0, 1] * u * v + A[1, 1] * v ** 2 - 2 * b[0] * u - 2 * b[1] * v
        assert 0 <= a1 <= 1 and 0 <= a2 <= 1
        assert float(q(torch.tensor(a1, dtype=torch.float64), torch.tensor(a2, dtype=torch.float64))) <= float(q(w1, w2).min()) + 1e-9


def test_box_ls_interior():
    A = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    w = torch.tensor([0.4, 0.7], dtype=torch.float64)
    b = A @ w
    a1, a2 = fs.box_ls(float(A[0, 0]), float(A[0, 1]), float(A[1, 1]), float(b[0]), float(b[1]))
    assert abs(a1 - 0.4) < 1e-12 and abs(a2 - 0.7) < 1e-12
