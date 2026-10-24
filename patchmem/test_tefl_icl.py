"""CPU tests for patchmem/r1_tefl_icl.py: demo indexing (no leak, query = T0-MS eps) and the ICL term (extra_delta on a stub: the
backbone needs the CUDA scan)."""
from types import SimpleNamespace
import torch
import torch.nn as nn
from patchmem.r1_tefl_icl import R1TeflICL, demos


def fake_E(b=2, T=20, K=3, C=2, P=4):
    """E[:, j, k] = the target patch index j+1+k everywhere (patch identity), avail as multistep_residuals (inside the lookback)."""
    j = torch.arange(T - 1)[:, None]
    tgt = j + 1 + torch.arange(K)[None]
    E = tgt[None, :, :, None, None].float().expand(b, T - 1, K, C, P).clone()
    avail = (tgt <= T - 1)[None].expand(b, T - 1, K).clone()
    return E * avail[..., None, None], avail


def test_demos_no_leak_and_query():
    T, K, P = 20, 3, 4
    E, avail = fake_E(T=T, K=K, P=P)
    eps, D = demos(E, avail, K)
    assert torch.equal(eps[0, 0].reshape(K, P)[:, 0], torch.tensor([T - K, T - K + 1, T - 1.]))   # query = last K observed patches
    n = T - 2 * K
    assert D['k'].shape[1] == n and bool(D['mask'].all())
    vt = D['v'][0, :, 0].reshape(n, K, P)[:, :, 0]                                                  # value target patches
    kt = D['k'][0, :, 0].reshape(n, K, P)[:, :, 0]
    assert vt.max() <= T - 1 and torch.equal(vt[:, 0], kt[:, -1] + 1)                                # value = the next horizon
    t0 = torch.tensor([0, 5])                                                                        # sample 1: origins < 5 invalid
    avail2 = avail & (torch.arange(T - 1)[None, :] >= t0[:, None])[:, :, None]
    _, D2 = demos(E * avail2[..., None, None], avail2, K)
    assert bool(D2['mask'][0].all()) and not bool(D2['mask'][1, :5].any()) and bool(D2['mask'][1, 5:].all())
    _, D3 = demos(*fake_E(T=2 * K, K=K), K)
    assert D3 is None                                                                                 # n = 0: no ICL term


def stub(variant, n=4, C=2, K=3, P=4, r=5, b=3):
    din = K * P
    s = SimpleNamespace(variant=variant, audit=None, hmax=K, icl_gen=torch.Generator().manual_seed(0),
                        W1=nn.Linear(din, r, bias=False), W2=nn.Linear(r, din, bias=False),
                        Wq=nn.Linear(din, r, bias=False), Wk=nn.Linear(din, r, bias=False), eta=nn.Parameter(torch.ones(K)))
    s.phi = lambda e, role: R1TeflICL.phi(s, e, role)
    g = torch.Generator().manual_seed(1)
    s.demo = dict(k=torch.randn(b, n, C, din, generator=g), v=torch.randn(b, n, C, din, generator=g), mask=torch.ones(b, n, dtype=torch.bool))
    delta, eps = torch.randn(b, C, K, P, generator=g), torch.randn(b, C, din, generator=g)
    return s, delta, eps


def test_id_one_demo_equal_key_returns_value():
    s, d, e = stub('id', n=1)
    s.demo['k'] = e[:, None].clone()
    out = R1TeflICL.extra_delta(s, d, e, None)
    assert torch.allclose(out.reshape(e.shape), s.demo['v'][:, 0], atol=1e-5)                     # W2 term dropped, eta * v


def test_formula_bounded_and_mask():
    for var in ('id', 'tied', 'qk'):
        s, d, e = stub(var)
        s.demo['mask'][0, 1:] = False
        out = R1TeflICL.extra_delta(s, d, e, None).reshape(e.shape)
        q, k = s.phi(e, 'q'), s.phi(s.demo['k'], 'k')
        m = s.demo['mask'][:, :, None].float()
        ref = ((k * q[:, None]).sum(-1)[..., None] * m[..., None] * s.demo['v']).sum(1) / (q.norm(dim=-1) * (k.norm(dim=-1) * m).sum(1) + 1e-6)[..., None]
        assert torch.allclose(out, ref, atol=1e-5), var
        assert (out.abs() <= s.demo['v'].abs().amax(1) + 1e-4).all(), var                         # bounded by the largest value


def test_hyb_starts_from_w2_and_offs():
    s, d, e = stub('hyb')
    s.eta.data.zero_()
    assert torch.equal(R1TeflICL.extra_delta(s, d, e, None), d)                                     # eta 0: TEFL delta unchanged
    s.eta.data.fill_(1.0)
    s.demo['v'] = s.W2(torch.relu(s.W1(s.demo['k']))).detach()                                       # demos fitted exactly by W2
    assert torch.allclose(R1TeflICL.extra_delta(s, d, e, None), d, atol=1e-5)                       # zero residual: no correction
    s2, d2, e2 = stub('tied')
    s2.audit = 'icloff'
    assert torch.equal(R1TeflICL.extra_delta(s2, d2, e2, None), torch.zeros_like(d2))               # tied: W2 off, ICL off -> 0


def test_iclperm_keeps_values_set_and_grads_finite():
    s, d, e = stub('tied', n=6)
    s.audit = 'iclperm'
    a = R1TeflICL.extra_delta(s, d, e, None)
    s.audit = None
    s.demo['v'] = s.demo['v'].flip(1)
    assert not torch.allclose(a, R1TeflICL.extra_delta(s, d, e, None))
    s3, d3, e3 = stub('tied')
    with torch.no_grad():
        s3.W1.weight.fill_(-1.0)                                                                     # ReLU features all zero
    e3 = e3.abs(); s3.demo['k'] = s3.demo['k'].abs()
    out = R1TeflICL.extra_delta(s3, d3, e3, None)
    out.sum().backward()
    assert torch.equal(out, torch.zeros_like(out)) and torch.isfinite(s3.W1.weight.grad).all() and torch.isfinite(s3.eta.grad).all()


def test_k_follow_padding_invariant():
    """K = 2 < Hmax = 3: zero-padding leads K+1..Hmax leaves the 'id' term unchanged on leads 1..K and zero beyond."""
    s, d, e = stub('id', n=3, K=3)
    KP = 2 * 4
    pad = lambda x: torch.cat([x[..., :KP], torch.zeros_like(x[..., KP:])], -1)
    s.demo['k'], s.demo['v'], e = pad(s.demo['k']), pad(s.demo['v']), pad(e)
    out = R1TeflICL.extra_delta(s, d, e, None).reshape(e.shape)
    q, k, v = e[..., :KP], s.demo['k'][..., :KP], s.demo['v'][..., :KP]
    ref = ((k * q[:, None]).sum(-1)[..., None] * v).sum(1) / (q.norm(dim=-1) * k.norm(dim=-1).sum(1) + 1e-6)[..., None]
    assert torch.allclose(out[..., :KP], ref, atol=1e-5) and torch.equal(out[..., KP:], torch.zeros_like(out[..., KP:]))
    E, avail = fake_E(T=15, K=2)
    _, D = demos(E, avail, 2)
    assert D['k'].shape[1] == 15 - 4                                                                 # L720 H96: 11 demos


def test_ridge_matches_primal():
    s, d, e = stub('ridge', n=4, K=3)
    s.demo['mask'][0, 2:] = False
    out = R1TeflICL.extra_delta(s, d, e, None).reshape(e.shape)
    import patchmem.r1_tefl_icl as m_
    for bi in range(e.shape[0]):
        for ci in range(e.shape[1]):
            mk = s.demo['mask'][bi]
            K, V = s.demo['k'][bi, mk, ci], s.demo['v'][bi, mk, ci]
            W = torch.linalg.solve(K.T @ K + m_.RIDGE_LAM * torch.eye(K.shape[1]), K.T @ V)
            assert torch.allclose(out[bi, ci], e[bi, ci] @ W, atol=1e-4)


if __name__ == '__main__':
    test_ridge_matches_primal()
    test_k_follow_padding_invariant()
    test_demos_no_leak_and_query(); test_id_one_demo_equal_key_returns_value(); test_formula_bounded_and_mask()
    test_hyb_starts_from_w2_and_offs(); test_iclperm_keeps_values_set_and_grads_finite(); print('ok')
