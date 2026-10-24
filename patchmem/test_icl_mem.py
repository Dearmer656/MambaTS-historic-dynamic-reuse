"""Tests for patchmem/icl_mem.py (CPU, synthetic): demo indexing, the 'last' / 'mean' kernels, eta and the shuffle null."""
import numpy as np
import torch
import patchmem.icl_mem as im

im.dev = 'cpu'


def stats_to_red(st, eta):
    s = st[0]
    return 1 - (s[2] - 2 * eta * s[0] + eta ** 2 * s[1]).sum().item() / s[2].sum().item()


def test_last_kernel_indexing():
    """own[i] = residual; with 'last' the correction for query q must be E[q - H/stride] exactly."""
    g = torch.Generator().manual_seed(0)
    N, C, r, sd = 60, 3, 4, 48
    own = torch.randn(N, im.H, C, generator=g)
    phi = torch.rand(N, C, r, generator=g)
    qi = np.arange(20, N)
    ds = list(range(im.H // sd, 3 * im.H // sd + 1, 1))
    # make the query residual equal to the demo d = H: own[q] = own[q - 7] for queries
    for q in qi:
        own[q] = own[q - im.H // sd]
    st = im.score(own, own.clone(), phi, qi, ds, torch.Generator().manual_seed(1), chunk=7)
    s = st[('own', 'last', 0)]
    assert torch.allclose(s[0, 0], s[0, 1]) and torch.allclose(s[0, 0], s[0, 2])        # d == r: <r,d> = <d,d> = <r,r>
    assert abs(stats_to_red(s, 1.0) - 1.0) < 1e-9


def test_softmax_selects_matching_key():
    """Values tied to keys: v_s = f(phi_s); query residual = f(phi_t). Sharp cos attention beats mean; shuffle breaks it."""
    g = torch.Generator().manual_seed(2)
    N, C, r = 200, 2, 6
    proto = torch.randn(4, r, generator=g).abs()
    vals = torch.randn(4, im.H, generator=g)
    lab = torch.randint(0, 4, (N, C), generator=g)
    phi = proto[lab] + 0.01 * torch.rand(N, C, r, generator=g)
    own = vals[lab].permute(0, 2, 1).contiguous()                                      # (N, H, C)
    qi = np.arange(50, N)
    ds = list(range(7, 7 + 40))
    st = im.score(own, own, phi, qi, ds, torch.Generator().manual_seed(3))
    ev = lambda k: im.evaluate(st[k], st[k])
    cosr, mean = ev(('own', 'cos', 0.03)), ev(('own', 'mean', 0))
    assert cosr['test'] > 0.9 and cosr['test'] > mean['test'] + 0.3, (cosr, mean)
    assert cosr['sh'] < cosr['test'] - 0.3, cosr


def test_eta_closed_form():
    g = torch.Generator().manual_seed(4)
    N, C, r = 80, 2, 3
    own = torch.randn(N, im.H, C, generator=g)
    phi = torch.rand(N, C, r, generator=g)
    st = im.score(own, own, phi, np.arange(30, N), list(range(7, 15)), torch.Generator().manual_seed(5))
    s = st[('own', 'mean', 0)]
    e = im.evaluate(s, s)
    grid = [stats_to_red(s, x) for x in np.linspace(e['eta'] - 1, e['eta'] + 1, 41)]
    assert e['val'] >= max(grid) - 1e-9


if __name__ == '__main__':
    test_last_kernel_indexing(); test_softmax_selects_matching_key(); test_eta_closed_form(); print('ok')
