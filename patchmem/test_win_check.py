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


if __name__ == '__main__':
    test_ridge_equals_stream_win(); test_controls_window_only(); test_control_maps(); print('ok')
