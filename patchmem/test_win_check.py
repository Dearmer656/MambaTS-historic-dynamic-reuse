"""CPU tests for patchmem/win_check.py: stream_ctrl('ridge') equals efc_cet.stream_win; every control uses only targets inside the
lookback [t-L, t); control maps behave as documented."""
import torch
import patchmem.efc_cet as ec
import patchmem.win_check as wc


def _E(N=70, C=2, seed=3):
    return torch.randn(N, 7, C, generator=torch.Generator().manual_seed(seed))


def test_ridge_equals_stream_win():
    E, H, L = _E(), 3, 16
    for feat in ('row', 'ced'):
        c1, a1 = ec.stream(E, H, 5, 2, True, feat, L)
        c2, a2 = wc.stream_ctrl(E, H, 5, 2, feat, L, 'ridge')
        assert torch.equal(a1, a2) and torch.allclose(c1, c2, atol=1e-4), (feat, (c1 - c2).abs().max())


def test_controls_window_only():
    E, H, L, t = _E(), 3, 16, 50
    E2 = E.clone()
    for i in range(E.shape[0]):
        for h in range(1, 8):
            if i + h - 1 >= t or i + h - 1 < t - L:
                E2[i, h - 1] += 5.0
    for md in wc.CTRLS:
        c1, a1 = wc.stream_ctrl(E, H, 0, 2, 'row', L, md, seed=1)
        c2, _ = wc.stream_ctrl(E2, H, 0, 2, 'row', L, md, seed=1)
        assert bool(a1[t]) and torch.allclose(c1[t], c2[t], atol=1e-5), md


def test_control_maps():
    g = torch.Generator().manual_seed(0)
    Fm, Ym, Fq = torch.randn(9, 2, 3, generator=g), torch.randn(9, 2, 3, generator=g), torch.randn(1, 2, 3, generator=g)
    assert torch.equal(wc.predict('persist', Fm, Ym, Fq, g), Fq)
    assert torch.allclose(wc.predict('bias', Fm, Ym, Fq, g)[0], Ym.mean(0))
    rho = (Fm * Ym).sum() / (ec.LAM + Fm.square().sum())
    assert torch.allclose(wc.predict('scalar', Fm, Ym, Fq, g), rho * Fq)
    # ridge_mean with constant targets returns the mean (centered targets are 0)
    Yc = torch.ones(9, 2, 3) * torch.tensor([1.0, -2.0]).view(1, 2, 1)
    assert torch.allclose(wc.predict('ridge_mean', Fm, Yc, Fq, g)[0], Yc[0], atol=1e-5)
    # shuffle keeps S = sum F^T F: with Y = F A and a permutation, the fit differs from the paired fit
    A = torch.randn(3, 3, generator=g); Ya = Fm @ A
    assert not torch.allclose(wc.predict('shuffle', Fm, Ya, Fq, g), wc.predict('ridge', Fm, Ya, Fq, g), atol=1e-3)


def test_attn_window_only():
    E, H, L, t = _E(), 3, 16, 50
    E2 = E.clone()
    for i in range(E.shape[0]):
        for h in range(1, 8):
            if i + h - 1 >= t or i + h - 1 < t - L:
                E2[i, h - 1] += 5.0
    for feat in ('row', 'ced'):
        for md in wc.ATTN:
            c1, a1 = wc.stream_ctrl(E, H, 0, 2, feat, L, md)
            c2, _ = wc.stream_ctrl(E2, H, 0, 2, feat, L, md)
            assert bool(a1[t]) and torch.allclose(c1[t], c2[t], atol=1e-5), (feat, md)


def test_attn_maps_explicit():
    """Each attention arm against a per-variable explicit (dual / loop) computation."""
    g = torch.Generator().manual_seed(1)
    Fm, Ym, Fq = torch.randn(12, 2, 3, generator=g), torch.randn(12, 2, 3, generator=g), torch.randn(2, 2, 3, generator=g)
    out = {md: wc.attn_predict(md, Fm, Ym, Fq) for md in wc.ATTN}
    for c in range(2):
        K = Fm[:, c] / Fm[:, c].norm(dim=-1, keepdim=True); Q = Fq[:, c] / Fq[:, c].norm(dim=-1, keepdim=True); Y = Ym[:, c]
        assert torch.allclose(out['lin1'][:, c], (Q @ K.T) @ Y / 12, atol=1e-5)
        G = (K @ K.T).double()
        dual_lin = (Q.double() @ K.T.double()) @ torch.linalg.solve(G + ec.LAM * torch.eye(12, dtype=torch.float64), Y.double())
        assert torch.allclose(out['krlin'][:, c].double(), dual_lin, atol=1e-4)            # primal H x H == dual N x N (linear kernel)
        d2 = [float(((K[i] - K[j]) ** 2).sum()) for i in range(12) for j in range(i + 1, 12)]
        s2 = float(torch.tensor(d2).median())
        kq = torch.exp(-((Q[:, None] - K[None]) ** 2).sum(-1) / (2 * s2))                   # median heuristic: exp(-d^2 / 2 s2) = exp((cos - 1) / s2)
        assert torch.allclose(out['soft1'][:, c], (kq / kq.sum(1, keepdim=True)) @ Y, atol=1e-5)
        assert torch.allclose(out['exp1'][:, c], kq @ Y / 12, atol=1e-5)
        Km = torch.exp(-((K[:, None] - K[None]) ** 2).sum(-1) / (2 * s2)).double()
        kr = kq.double() @ torch.linalg.solve(Km + ec.LAM * torch.eye(12, dtype=torch.float64), Y.double())
        assert torch.allclose(out['krexp'][:, c].double(), kr, atol=1e-4)


def test_res_stage2():
    """vals = E reproduces the plain run; scale divides the bandwidth; stage 2 sees only targets inside the lookback (E and residuals)."""
    E, H, L, t = _E(), 3, 16, 50
    c1, _ = wc.stream_ctrl(E, H, 0, 2, 'row', L, 'soft1')
    c2, _ = wc.stream_ctrl(E, H, 0, 2, 'row', L, 'soft1', vals=E.clone())
    assert torch.allclose(c1, c2)
    g = torch.Generator().manual_seed(4)
    Fm, Ym, Fq = torch.randn(12, 2, 3, generator=g), torch.randn(12, 2, 3, generator=g), torch.randn(1, 2, 3, generator=g)
    K = Fm[:, 0] / Fm[:, 0].norm(dim=-1, keepdim=True); Q = Fq[:, 0] / Fq[:, 0].norm(dim=-1, keepdim=True)
    s2 = float(wc._sigma2(K[None]))
    w = torch.softmax((Q @ K.T - 1) / (s2 / 4), dim=-1)
    assert torch.allclose(wc.attn_predict('soft1', Fm, Ym, Fq, 0.25)[:, 0], w @ Ym[:, 0], atol=1e-5)
    R = E[:, :H] - 0.3 * torch.roll(E[:, :H], 1, 0)
    E2, R2 = E.clone(), R.clone()
    for i in range(E.shape[0]):
        for h in range(1, 8):
            if i + h - 1 >= t or i + h - 1 < t - L:
                E2[i, h - 1] += 5.0
                if h <= H:
                    R2[i, h - 1] += 5.0
    a, act = wc.stream_ctrl(E, H, 0, 2, 'row', L, 'soft1', vals=R, scale=0.25)
    b, _ = wc.stream_ctrl(E2, H, 0, 2, 'row', L, 'soft1', vals=R2, scale=0.25)
    assert bool(act[t]) and torch.allclose(a[t], b[t], atol=1e-5)


if __name__ == '__main__':
    test_ridge_equals_stream_win(); test_controls_window_only(); test_control_maps(); test_attn_window_only(); test_attn_maps_explicit(); test_res_stage2(); print('ok')
