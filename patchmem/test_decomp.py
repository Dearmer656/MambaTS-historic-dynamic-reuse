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
    return dc.gd_weights(k, dc.prefix(S), dc.prefix(G), w0, torch.tensor(ages), h, grid)


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


def ref_carry(rho, sig, T0, h, pw, nb, gamma):
    """A = I, lam = 0 reference: each written slot holds the latest write; dict-based."""
    n, B, _ = rho.shape
    state = torch.zeros(B, nb, pw)
    out = []
    for k in range(n):
        t = T0 + k
        if k - h >= 0:
            acc = {}
            for kk in range(h):
                key = (kk * nb // h, (t - h + kk) % pw)
                acc.setdefault(key, []).append(rho[k - h][:, kk] / sig[k - h] ** gamma)
            for (b_, p_), v in acc.items():
                state[:, b_, p_] = torch.stack(v).mean(0)
        out.append(torch.stack([state[:, hh * nb // h, (t + hh) % pw] for hh in range(h)], 1) * sig[k][:, None] ** gamma)
    return torch.stack(out)


def _run(net, rho, x, sig, T0, h):
    c, out = None, []
    with torch.no_grad():
        for k in range(len(rho)):
            c, m = dc.carry_step(net, c, k, T0, lambda q: rho[q], lambda q: x[q], lambda q: sig[q], h)
            out.append(m)
    return torch.stack(out)


def test_write_read_alignment():
    """A = I, lam -> 0: m equals the dict reference (slot = (lead bucket, target hour), latest write, sigma gain)."""
    for h, pw, nb, gamma in ((8, 6, 2, 0.0), (8, 6, 2, 1.0), (12, 5, 3, 1.0), (6, 4, 1, 0.5)):
        B, T0, n = 3, 50, 40
        net = dc.Carry(2, full=False, pw=pw, nb=nb)
        with torch.no_grad():
            net.wl.weight.zero_()
            net.wl.bias.fill_(-50.0)
            net.gamma.fill_(gamma)
        g = torch.Generator().manual_seed(1)
        rho = torch.randn(n, B, h, generator=g)
        x = torch.randn(n, B, 2, generator=g)
        sig = torch.rand(n, B, generator=g) + 0.5
        assert torch.allclose(_run(net, rho, x, sig, T0, h), ref_carry(rho, sig, T0, h, pw, nb, gamma), atol=1e-5)


def test_carry_causal():
    """full carry: m at index k does not depend on rho of indices > k - h."""
    h, pw, nb, B, T0, n = 5, 7, 2, 2, 30, 30
    torch.manual_seed(0)
    net = dc.Carry(4, full=True, pw=pw, nb=nb)
    g = torch.Generator().manual_seed(2)
    rho = torch.randn(n, B, h, generator=g)
    x = torch.randn(n, B, 4, generator=g)
    sig = torch.rand(n, B, generator=g) + 0.5
    k = 20
    r2 = rho.clone()
    r2[k - h + 1:] += 4.0
    a, b = _run(net, rho, x, sig, T0, h), _run(net, r2, x, sig, T0, h)
    assert torch.equal(a[:k + 1], b[:k + 1])
    assert not torch.equal(a[k + 1:], b[k + 1:])


def test_carry_time_equivariant():
    """the Householder direction is relative to the current hour: shifting t by 1 and rolling state / writes by 1
    along the hour axis rolls the new state by 1 along the hour axis."""
    pw, nb, B = 7, 3, 3
    torch.manual_seed(0)
    net = dc.Carry(4, full=True, pw=pw, nb=nb)
    g = torch.Generator().manual_seed(3)
    c, x, w = torch.randn(B, nb * pw, generator=g), torch.randn(B, 4, generator=g), torch.randn(B, nb * pw, generator=g)
    wm = torch.rand(nb * pw, generator=g) > 0.5
    r = lambda v: v.view(*v.shape[:-1], nb, pw).roll(1, -1).flatten(-2)
    with torch.no_grad():
        a, _, _ = net.step(c, x, 10, w, wm)
        b, _, _ = net.step(r(c), x, 11, r(w), r(wm))
    assert torch.allclose(r(a), b, atol=1e-6)


def test_phi_clip_and_onehot():
    """z features are clipped to +-Z_CLIP even for near-constant channels; one-hot hour / day blocks sum to 1."""
    b, C, h = 2, 3, 30
    d = torch.full((b, C, h), 1e6)
    norm = (torch.zeros(5, C), torch.full((5, C), dc.SD_Z))
    P = dc.phi(d, -d, d, d[..., 0], -d[..., 0], torch.tensor([7, 500]), norm)
    assert P.shape[-1] == dc.NP
    assert P[..., 1:6].abs().max() <= dc.Z_CLIP
    assert torch.equal(P[..., 6:30].sum(-1), torch.ones(b, C, h)) and torch.equal(P[..., 30:].sum(-1), torch.ones(b, C, h))
