import torch

import patchmem.eval_decomp as dc


def rand_case(n=80, C=2, h=4, p=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(n, C, h, p, generator=g, dtype=torch.float64)
    u = torch.randn(n, C, h, generator=g, dtype=torch.float64)
    return phi, u


def weights(phi, u, ages, h, grid, w0):
    S, G = dc.demo_sums(phi, u)
    k = torch.arange(len(phi))
    return dc.gd_weights(k, S, G, dc.prefix(S), dc.prefix(G), w0, torch.tensor(ages), h, grid)


def test_gd_matches_direct():
    """vectorised two-step GD == explicit demo loops: periodic step from w0, then recent step from w1."""
    phi, u = rand_case()
    h, ages = 4, [4, 6, 9]
    grid = [(0.5, 1.0, 5), (1.0, 0.0, 24), (0.0, 1.5, 3), (1.5, 0.25, 50)]
    w0 = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    W = weights(phi, u, ages, h, grid, w0)

    def step(w, idx, eta, c):
        if not idx or eta == 0:
            return w
        g = sum(phi[j, c].T @ (u[j, c] - phi[j, c] @ w) for j in idx)
        tr = sum((phi[j, c] ** 2).sum() for j in idx)
        return w + eta * g / tr
    for k in (0, 3, 4, 9, 30, 79):
        for e1, e2, N in grid:
            for c in range(2):
                per = [k - a for a in ages if k - a >= 0]
                rec = [j for j in range(k - h - N + 1, k - h + 1) if j >= 0]
                w1 = step(w0, per, e1, c)
                w2 = step(w1, rec, e2, c)
                assert torch.allclose(W[e1, e2, N][k, c], w2, atol=1e-10), (k, e1, e2, N, c)


def test_gd_causal():
    """w(k) does not depend on u of stream indices > k - h."""
    phi, u = rand_case()
    h, ages, grid = 4, [4, 6, 9], [(1.0, 1.0, 10)]
    w0 = torch.zeros(3, dtype=torch.float64)
    k = 40
    W = weights(phi, u, ages, h, grid, w0)
    u2 = u.clone()
    u2[k - h + 1:] += 5.0
    phi2 = phi.clone()
    phi2[k - h + 1:] += 3.0
    W2 = weights(phi2, u2, ages, h, grid, w0)
    assert torch.equal(W[grid[0]][:k + 1], W2[grid[0]][:k + 1])


def test_per_ages():
    a = dc.per_ages(192)
    assert min(a) >= 192 and a[:2] == [192, 216] and 336 in a and len(a) == len(set(a)) == 15


def test_phase_features_lead_specific():
    """phase features of (t, h + 1) equal those of (t + 1, h): they index the target time t + h."""
    b, C, h = 2, 3, 6
    z = torch.zeros(b, C, h)
    norm = (torch.zeros(5, C), torch.ones(5, C))
    t = torch.tensor([100, 101])
    P = dc.phi(z, z, z, z[..., 0], z[..., 0], t, norm)
    assert torch.allclose(P[0, :, 1:, 6:], P[1, :, :-1, 6:], atol=1e-6)
    assert not torch.allclose(P[0, :, 0, 6:], P[0, :, 1, 6:])


def test_write_read_alignment():
    """A = I, lam -> 0, gamma = 0: m at index k is the slot mean of rho[k - h] at the target slot of each lead."""
    h, pw, B, T0 = 8, 6, 3, 50
    net = dc.Carry(2, full=False, pw=pw)
    with torch.no_grad():
        net.wl.weight.zero_()
        net.wl.bias.fill_(-50.0)
        net.gamma.fill_(0.0)
    g = torch.Generator().manual_seed(1)
    rho = torch.randn(40, B, h, generator=g)
    x = torch.randn(40, B, 2, generator=g)
    sig = torch.rand(40, B, generator=g) + 0.5
    c = None
    for k in range(40):
        c, m = dc.carry_step(net, c, k, T0, lambda q: rho[q], lambda q: x[q], lambda q: sig[q], h)
        if k < h:
            assert torch.equal(m, torch.zeros_like(m))
            continue
        t, src = T0 + k, T0 + k - h
        for hh in range(h):
            leads = [kk for kk in range(h) if (src + kk) % pw == (t + hh) % pw]
            want = rho[k - h][:, leads].mean(1)
            assert torch.allclose(m[:, hh], want, atol=1e-5), (k, hh)


def test_carry_causal():
    """full carry: m at index k does not depend on rho of indices > k - h."""
    h, pw, B, T0, n = 5, 7, 2, 30, 30
    torch.manual_seed(0)
    net = dc.Carry(4, full=True, pw=pw)
    g = torch.Generator().manual_seed(2)
    rho = torch.randn(n, B, h, generator=g)
    x = torch.randn(n, B, 4, generator=g)
    sig = torch.rand(n, B, generator=g) + 0.5

    def run(r):
        c, out = None, []
        with torch.no_grad():
            for k in range(n):
                c, m = dc.carry_step(net, c, k, T0, lambda q: r[q], lambda q: x[q], lambda q: sig[q], h)
                out.append(m)
        return torch.stack(out)
    k = 20
    r2 = rho.clone()
    r2[k - h + 1:] += 4.0
    assert torch.equal(run(rho)[:k + 1], run(r2)[:k + 1])
    assert not torch.equal(run(rho)[k + 1:], run(r2)[k + 1:])


def test_carry_time_equivariant():
    """the Householder direction is relative to the current hour: shifting t by 1 and rolling state / writes by 1
    rolls the new state by 1."""
    pw, B = 7, 3
    torch.manual_seed(0)
    net = dc.Carry(4, full=True, pw=pw)
    g = torch.Generator().manual_seed(3)
    c, x, w = torch.randn(B, pw, generator=g), torch.randn(B, 4, generator=g), torch.randn(B, pw, generator=g)
    wm = torch.rand(pw, generator=g) > 0.5
    with torch.no_grad():
        a, _, _ = net.step(c, x, 10, w, wm)
        b, _, _ = net.step(c.roll(1, -1), x, 11, w.roll(1, -1), wm.roll(1, -1))
    assert torch.allclose(a.roll(1, -1), b, atol=1e-6)


def test_gamma_gain():
    """gamma = 1, lam -> 0, A = I: m = rho[k - h] * sig_k / sig_{k-h} on single-lead slots."""
    h, pw, B, T0 = 4, 6, 2, 0
    net = dc.Carry(2, full=False, pw=pw)
    with torch.no_grad():
        net.wl.weight.zero_()
        net.wl.bias.fill_(-50.0)
    g = torch.Generator().manual_seed(4)
    rho = torch.randn(12, B, h, generator=g)
    x = torch.randn(12, B, 2, generator=g)
    sig = torch.rand(12, B, generator=g) + 0.5
    c = None
    for k in range(12):
        c, m = dc.carry_step(net, c, k, T0, lambda q: rho[q], lambda q: x[q], lambda q: sig[q], h)
    k = 11
    # lead hh of origin k targets slot (k + hh) % pw, written now by source lead kk = (hh + h) % pw if kk < h
    hit = 0
    for hh in range(h):
        kk = (hh + h) % pw
        if kk >= h:
            continue
        hit += 1
        want = rho[k - h][:, kk] * sig[k] / sig[k - h]
        assert torch.allclose(m[:, hh], want, atol=1e-5)
    assert hit == 2
