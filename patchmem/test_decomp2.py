import torch

import patchmem.eval_decomp as dc
import patchmem.eval_decomp2 as d2


def test_topk_read_matches_brute_force():
    """(1/k) sum over the top-k cos among indices <= q - h of cos * E, fewer candidates -> sum of all / k."""
    g = torch.Generator().manual_seed(0)
    c, n, l, h = 3, 60, 7, 5
    K = torch.randn(c, n, l, generator=g, dtype=torch.float64)
    K = K / K.norm(dim=-1, keepdim=True)
    E = torch.randn(n, c, h, generator=g, dtype=torch.float64)
    ks = (1, 4, 30)
    kq = torch.tensor([0, 4, 5, 8, 20, 59])
    out = d2.topk_read(K, E, kq, h, ks)
    for bi, q in enumerate(kq.tolist()):
        for ci in range(c):
            cand = list(range(0, q - h + 1))
            cos = torch.tensor([float(K[ci, j] @ K[ci, q]) for j in cand], dtype=torch.float64)
            order = sorted(range(len(cand)), key=lambda i: -float(cos[i]))
            for k in ks:
                sel = order[:k]
                want = sum((cos[i] * E[cand[i], ci] for i in sel), torch.zeros(h, dtype=torch.float64)) / k
                assert torch.allclose(out[k][bi, ci], want, atol=1e-10), (q, ci, k)


def test_diagonal():
    g = torch.Generator().manual_seed(1)
    R = torch.randn(20, 2, 6, generator=g)
    D = d2.diagonal(R)
    for k in range(20):
        for l in range(6):
            want = R[k - 1 - l, :, l] if k - 1 - l >= 0 else torch.zeros(2)
            assert torch.equal(D[k, :, l], want)


def ref_diag(R, sig, T0, h, pw, nb, gamma):
    """A = I, lam = 0: slot (bucket(l), (t-1) % pw) holds the latest mean of the valid diagonal entries / sig_t^g."""
    n, B, _ = R.shape
    state = torch.zeros(B, nb, pw)
    out = []
    for k in range(n):
        t = T0 + k
        acc = {}
        for l in range(h):
            if k - 1 - l >= 0:
                acc.setdefault(l * nb // h, []).append(R[k - 1 - l][:, l] / sig[k] ** gamma)
        for b_, v in acc.items():
            state[:, b_, (t - 1) % pw] = torch.stack(v).mean(0)
        out.append(torch.stack([state[:, hh * nb // h, (t + hh) % pw] for hh in range(h)], 1) * sig[k][:, None] ** gamma)
    return torch.stack(out)


def run_carry(net, R, x, sig, T0, h, mode):
    D = d2.diagonal(R)
    src = D if mode == 'diag' else R
    c, out = None, []
    with torch.no_grad():
        for k in range(len(R)):
            c, m = d2.carry_step2(net, c, k, T0, mode, lambda q: src[q], lambda q: x[q], lambda q: sig[q], h)
            out.append(m)
    return torch.stack(out)


def test_diag_write_read_reference():
    for h, pw, nb, gamma in ((8, 6, 2, 0.0), (8, 6, 2, 1.0), (12, 5, 3, 0.7)):
        B, T0, n = 3, 41, 40
        net = dc.Carry(2, full=False, pw=pw, nb=nb)
        with torch.no_grad():
            net.wl.weight.zero_()
            net.wl.bias.fill_(-50.0)
            net.gamma.fill_(gamma)
        g = torch.Generator().manual_seed(2)
        R = torch.randn(n, B, h, generator=g)
        x = torch.randn(n, B, 2, generator=g)
        sig = torch.rand(n, B, generator=g) + 0.5
        assert torch.allclose(run_carry(net, R, x, sig, T0, h, 'diag'), ref_diag(R, sig, T0, h, pw, nb, gamma),
                              atol=1e-5)


def test_diag_carry_causal():
    """m at index k only uses errors of targets already observed: R[s, l] with s + l <= k - 1."""
    h, pw, nb, B, T0, n = 5, 7, 2, 2, 30, 30
    torch.manual_seed(0)
    net = dc.Carry(4, full=True, pw=pw, nb=nb)
    g = torch.Generator().manual_seed(3)
    R = torch.randn(n, B, h, generator=g)
    x = torch.randn(n, B, 4, generator=g)
    sig = torch.rand(n, B, generator=g) + 0.5
    k = 18
    s = torch.arange(n)[:, None, None]
    l = torch.arange(h)[None, None]
    R2 = torch.where(s + l >= k, R + 3.0, R)
    a, b = run_carry(net, R, x, sig, T0, h, 'diag'), run_carry(net, R2, x, sig, T0, h, 'diag')
    assert torch.equal(a[:k + 1], b[:k + 1])
    assert not torch.equal(a[k + 1:], b[k + 1:])


def test_full_mode_matches_v2_step():
    """mode 'full' reproduces eval_decomp.carry_step."""
    h, pw, nb, B, T0, n = 6, 5, 2, 2, 12, 20
    torch.manual_seed(1)
    net = dc.Carry(3, full=True, pw=pw, nb=nb)
    g = torch.Generator().manual_seed(4)
    R = torch.randn(n, B, h, generator=g)
    x = torch.randn(n, B, 3, generator=g)
    sig = torch.rand(n, B, generator=g) + 0.5
    a = run_carry(net, R, x, sig, T0, h, 'full')
    c, out = None, []
    with torch.no_grad():
        for k in range(n):
            c, m = dc.carry_step(net, c, k, T0, lambda q: R[q], lambda q: x[q], lambda q: sig[q], h)
            out.append(m)
    assert torch.allclose(a, torch.stack(out))


def test_norm_windows():
    X = torch.randn(50, 4)
    t = torch.tensor([10, 30])
    c = torch.tensor([1, 3])
    w = d2.norm_windows(X, t, c, 6)
    for ci, cc in enumerate(c.tolist()):
        for ti, tt in enumerate(t.tolist()):
            raw = X[tt - 6:tt, cc]
            want = (raw - raw.mean()) / torch.sqrt(raw.var(unbiased=False) + 1e-5)
            assert torch.allclose(w[ci, ti], want, atol=1e-5)
