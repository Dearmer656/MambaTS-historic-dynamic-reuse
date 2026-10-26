"""CPU tests for patchmem/r1_tefl_icl.py: demo indexing (no leak, query = T0-MS eps) and the ICL term (extra_delta on a stub: the
backbone needs the CUDA scan)."""
from types import SimpleNamespace
import torch
import torch.nn as nn
from patchmem.r1_tefl_icl import R1TeflICL, demos, demos_ced, permute_valid, lb_errors, lb_pad


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
    s = SimpleNamespace(variant=variant, audit=None, hmax=K, ridge_rel=None, nshrink=0.0, icl_gen=torch.Generator().manual_seed(0),
                        perm_gen=torch.Generator().manual_seed(1), wlog=None, win_ptr=0, case_collect=False, case_period=6, case_bank=True,
                        _bank=[], _icl_log=[], bank=None, icl_mean=None, sel_phase=[3, 5], sel_log=[], K_last=K,
                        SEL_SETS=R1TeflICL.SEL_SETS, sel_wdump=None, selw_cache={},
                        W1=nn.Linear(din, r, bias=False), W2=nn.Linear(r, din, bias=False),
                        Wq=nn.Linear(din, r, bias=False), Wk=nn.Linear(din, r, bias=False), eta=nn.Parameter(torch.ones(K)))
    s.phi = lambda e, role: R1TeflICL.phi(s, e, role)
    s.case_transform = lambda a: R1TeflICL.case_transform(s, a)
    s.case_basis = lambda kind, n, dev: R1TeflICL.case_basis(s, kind, n, dev)
    s.finalize_case = lambda: R1TeflICL.finalize_case(s)
    s.sel_masks = lambda *a: R1TeflICL.sel_masks(s, *a)
    s.sel_fit = lambda *a: R1TeflICL.sel_fit(s, *a)
    s.sel_correct = lambda *a, **kw: R1TeflICL.sel_correct(s, *a, **kw)
    s.two_layer = lambda *a, **kw: R1TeflICL.two_layer(s, *a, **kw)
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


def test_demos_ced_anti_diagonal():
    """key='ced': every key entry targets the same patch (query: the last observed patch T-1; demo o: patch o), values target the
    next K patches o+1..o+K <= T-1, n = T-2K as 'row', masks follow padded origins, and the query is never a demo."""
    T, K, P = 20, 3, 4
    E, avail = fake_E(T=T, K=K, P=P)
    eps, D = demos_ced(E, avail, K)
    assert torch.equal(eps[0, 0], torch.full((K * P,), T - 1.))
    n = T - 2 * K
    o = torch.arange(K, T - K).float()
    kt = D['k'][0, :, 0].reshape(n, K, P)[:, :, 0]
    vt = D['v'][0, :, 0].reshape(n, K, P)[:, :, 0]
    assert D['k'].shape[1] == n and bool(D['mask'].all())
    assert torch.equal(kt, o[:, None].expand(n, K)) and torch.equal(vt, o[:, None] + 1 + torch.arange(K)[None]) and vt.max() <= T - 1
    t0 = torch.tensor([0, 6])                                                                        # sample 1: origins < 6 invalid
    avail2 = avail & (torch.arange(T - 1)[None, :] >= t0[:, None])[:, :, None]
    _, D2 = demos_ced(E * avail2[..., None, None], avail2, K)
    ok = (o - K >= 6)                                                                                # earliest key origin o-K valid
    assert bool(D2['mask'][0].all()) and torch.equal(D2['mask'][1], ok)
    assert demos_ced(*fake_E(T=2 * K, K=K), K)[1] is None


def test_ridge_rel_matches_primal():
    """ridge_rel = s: lam = s * mean |k_j|^2 over valid demos."""
    s, d, e = stub('ridge', n=4, K=3)
    s.ridge_rel = 10.0
    s.demo['mask'][0, 2:] = False
    out = R1TeflICL.extra_delta(s, d, e, None).reshape(e.shape)
    for bi in range(e.shape[0]):
        for ci in range(e.shape[1]):
            mk = s.demo['mask'][bi]
            K, V = s.demo['k'][bi, mk, ci], s.demo['v'][bi, mk, ci]
            lam = 10.0 * K.square().sum(-1).mean()
            W = torch.linalg.solve(K.T @ K + lam * torch.eye(K.shape[1]), K.T @ V)
            assert torch.allclose(out[bi, ci], e[bi, ci] @ W, atol=1e-4)


def test_hybridge_prior_anchored():
    """hybridge: delta (= TEFL's W2 phi(q), computed by the caller) + (W - W2) phi(q), W = primal argmin sum |v - W phi|^2 +
    lam |W - W2|^2; lam -> inf and icloff leave delta (TEFL)."""
    s, d, e = stub('hybridge', n=4, K=3)
    s.ridge_rel = 1.0
    s.demo['mask'][1, 3:] = False
    out = R1TeflICL.extra_delta(s, d, e, None).reshape(e.shape)
    W2 = s.W2.weight.detach()                                                                        # (din, r)
    with torch.no_grad():
        for bi in range(e.shape[0]):
            for ci in range(e.shape[1]):
                mk = s.demo['mask'][bi]
                Ph = torch.relu(s.W1(s.demo['k'][bi, mk, ci])); V = s.demo['v'][bi, mk, ci]; pq = torch.relu(s.W1(e[bi, ci]))
                lam = Ph.square().sum(-1).mean()
                W = torch.linalg.solve(Ph.T @ Ph + lam * torch.eye(Ph.shape[1]), Ph.T @ V + lam * W2.T).T   # (din, r)
                assert torch.allclose(out[bi, ci] - d.reshape(e.shape)[bi, ci], (W - W2) @ pq, atol=1e-4)
        s.ridge_rel = 1e8
        out = R1TeflICL.extra_delta(s, d, e, None)
        assert torch.allclose(out, d, atol=1e-4)
    s.audit = 'icloff'
    assert torch.equal(R1TeflICL.extra_delta(s, d, e, None), d)


def test_nshrink():
    s, d, e = stub('id', n=4, K=3)
    s.demo['mask'][0, 1:] = False                                                                    # sample 0: 1 demo, others 4
    base = R1TeflICL.extra_delta(s, d, e, None)
    s.nshrink = 8.0
    out = R1TeflICL.extra_delta(s, d, e, None)
    assert torch.allclose(out[0], base[0] / 9, atol=1e-6) and torch.allclose(out[1], base[1] * 4 / 12, atol=1e-6)


if __name__ == '__main__':
    test_ridge_rel_matches_primal(); test_hybridge_prior_anchored(); test_nshrink()
    test_demos_ced_anti_diagonal()
    test_ridge_matches_primal()
    test_k_follow_padding_invariant()
    test_demos_no_leak_and_query(); test_id_one_demo_equal_key_returns_value(); test_formula_bounded_and_mask()
    test_hyb_starts_from_w2_and_offs(); test_iclperm_keeps_values_set_and_grads_finite(); print('ok')


def test_hybmean_query_free_and_formula():
    s, d, e = stub('hybmean')
    s.demo['mask'][0, 2:] = False
    out = R1TeflICL.extra_delta(s, d, e, None)
    k = torch.relu(s.W1(s.demo['k'])); r = s.demo['v'] - s.W2(k)
    m = s.demo['mask'][:, :, None].float()
    w = k.norm(dim=-1) * m                                                                          # |phi(k_j)| on valid demos
    ref = (w[..., None] * r).sum(1) / (w.sum(1)[..., None] + 1e-6)
    assert torch.allclose((out - d).reshape(e.shape), ref, atol=1e-4)
    out2 = R1TeflICL.extra_delta(s, d, torch.randn_like(e) * 3, None)
    assert torch.allclose(out, out2, atol=1e-4)                                                     # no query dependence


def test_hybperm_breaks_pairing_keeps_set():
    g = torch.Generator().manual_seed(5)
    v = torch.arange(2 * 6).float().view(2, 6, 1, 1).expand(2, 6, 2, 3).clone()
    mask = torch.ones(2, 6, dtype=torch.bool); mask[1, 4:] = False
    pv = permute_valid(v, mask, g)
    assert torch.equal(pv[0, :, 0, 0].sort().values, v[0, :, 0, 0])                                 # same multiset
    assert torch.equal(pv[1, 4:], v[1, 4:]) and torch.equal(pv[1, :4, 0, 0].sort().values, v[1, :4, 0, 0])   # invalid rows fixed
    assert not torch.equal(pv, v)
    s, d, e = stub('hybperm', n=1)
    s1, _, _ = stub('hyb', n=1)
    s1.W1, s1.W2, s1.demo = s.W1, s.W2, s.demo
    assert torch.allclose(R1TeflICL.extra_delta(s, d, e, None), R1TeflICL.extra_delta(s1, d, e, None))   # 1 demo: nothing to permute
    s, d, e = stub('hybperm', n=6)
    s1, _, _ = stub('hyb', n=6)
    s1.W1, s1.W2, s1.demo = s.W1, s.W2, s.demo
    assert not torch.allclose(R1TeflICL.extra_delta(s, d, e, None), R1TeflICL.extra_delta(s1, d, e, None))


def test_wdump_records_weights_without_changing_output():
    s, d, e = stub('hyb', n=5)
    a = R1TeflICL.extra_delta(s, d, e, None)
    s.wlog = []
    b = R1TeflICL.extra_delta(s, d, e, None)
    assert torch.equal(a, b) and len(s.wlog) == 1 and s.wlog[0].shape == (3, 5)
    q, k = torch.relu(s.W1(e)), torch.relu(s.W1(s.demo['k']))
    w = (k * q[:, None]).sum(-1) / (q.norm(dim=-1)[:, None] * k.norm(dim=-1).sum(1)[:, None] + 1e-6)
    assert torch.allclose(s.wlog[0], w.mean(-1), atol=1e-5)
    s.audit = 'iclperm'
    R1TeflICL.extra_delta(s, d, e, None)
    assert len(s.wlog) == 1                                                                         # audits are not recorded


def _run(s, d, e, audit=None):
    s.audit = audit
    return R1TeflICL.extra_delta(s, d, e, None)


def test_case_collect_const_and_swap():
    """normal pass banks demos + logs Delta; iclconst = the mean Delta; iclswap0 = normal; iclswap1 uses the previous window's demos."""
    s, d, e = stub('hyb', n=4, b=3)
    s.case_collect = True; s.win_ptr = 0
    out = _run(s, d, e)
    icl = R1TeflICL.finalize_case(s)
    s.case_collect = False
    assert s.bank[0].shape == (3, 4, 2, 12) and icl.shape == (3, 2, 12)
    assert torch.allclose((out - d).reshape(3, 2, 12), icl, atol=1e-5)
    s.win_ptr = 0
    const = _run(s, d, e, 'iclconst')
    assert torch.allclose((const - d).reshape(3, 2, 12), icl.float().mean(0, keepdim=True).expand(3, 2, 12), atol=1e-5)
    s.win_ptr = 0
    assert torch.allclose(_run(s, d, e, 'iclswap0'), out, atol=1e-5)                                # fp32 bank: exact
    s.win_ptr = 0
    sw = _run(s, d, e, 'iclswap1')
    D0 = dict(s.demo)
    s.demo = dict(k=D0['k'][[2, 0, 1]], v=D0['v'][[2, 0, 1]], mask=D0['mask'][[2, 0, 1]])           # window i gets window i-1 (mod 3)
    s.win_ptr = 0
    ref = _run(s, d, e)
    s.demo = D0
    assert torch.allclose(sw, ref, atol=1e-5) and s.win_ptr == 3


def test_case_projections():
    s, _, _ = stub('hyb')
    t = torch.arange(12.)
    lin = (2 + 0.5 * t)[None, None].expand(3, 2, 12)
    sinus = torch.sin(2 * torch.pi * t / 6)[None, None].expand(3, 2, 12)
    s.audit = 'iclproj_const'
    assert torch.allclose(R1TeflICL.case_transform(s, lin), lin.mean(-1, keepdim=True).expand_as(lin), atol=1e-4)
    s.audit = 'iclproj_lin'
    assert torch.allclose(R1TeflICL.case_transform(s, lin), lin, atol=1e-4)
    assert R1TeflICL.case_transform(s, sinus).abs().max() < 0.6                                       # not in span(const, lin)
    s.audit = 'iclproj_daily'
    assert torch.allclose(R1TeflICL.case_transform(s, lin + sinus), lin + sinus, atol=1e-4)
    x = torch.randn(3, 2, 12)
    s.audit = 'iclproj_common'
    assert torch.allclose(R1TeflICL.case_transform(s, x), x.mean(1, keepdim=True).expand_as(x))
    s.audit = None
    assert torch.equal(R1TeflICL.case_transform(s, x), x)


def test_sel_masks_and_fit():
    s, d, e = stub('hyb', n=10, b=2)                                                                 # K = 3, query origin j0 = n + K - 1 = 12
    q = torch.relu(s.W1(e)); k = torch.relu(s.W1(s.demo['k'])); v = s.demo['v'] - s.W2(k)
    Phi, R = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    valid = torch.ones(2, 10, dtype=torch.bool)
    sets, base = R1TeflICL.sel_masks(s, Phi, q, 12, valid)
    assert bool(base.all())                                                                         # every demo has lag >= K at j0
    lag = 12 - torch.arange(10)
    assert torch.equal(sets['recent4'][0, 0], (lag >= 3) & (lag < 7))
    assert torch.equal(sets['phase'][0, 0], (lag == 3) | (lag == 5))
    assert int(sets['topk8'][0, 0].sum()) == 8
    sp, bp = R1TeflICL.sel_masks(s, Phi, Phi[:, :, 6], 6, valid)                                    # pseudo-query at demo 6: j <= 3 usable
    assert torch.equal(bp[0, 0], torch.arange(10) <= 3)
    # ridge on a subset == explicit per-(b, c) dual ridge
    M = sets['recent4']
    out = R1TeflICL.sel_fit(s, Phi, R, q, M, base, False, 3, 4)
    for bi in range(2):
        for ci in range(2):
            idx = M[bi, ci].nonzero().squeeze(-1)
            F, Y = Phi[bi, ci, idx].double(), R[bi, ci, idx].double()
            lam = Phi[bi, ci].square().sum(-1).mean().double()
            ref = Y.T @ torch.linalg.solve(F @ F.T + lam * torch.eye(len(idx), dtype=torch.float64), F @ q[bi, ci].double())
            assert torch.allclose(out[bi, ci].double(), ref, atol=1e-4)
    # one GD step on all demos == hyb's own ICL term
    add = R1TeflICL.sel_fit(s, Phi, R, q, base, base, True, 3, 4)
    full = (R1TeflICL.extra_delta(s, d, e, None) - d).reshape(2, 2, 12)
    assert torch.allclose(add, full, atol=1e-5)


def test_iclsel_modes_run():
    s, d, e = stub('hyb', n=10, b=2)
    for mode in ('iclsel', 'iclsel1', 'iclselfix_phase'):
        s.win_ptr = 0
        out = _run(s, d, e, mode)
        assert out.shape == d.shape and torch.isfinite(out).all() and s.win_ptr == 2, mode
    assert len(s.sel_log) == 2 and torch.allclose(s.sel_log[0].sum(), torch.tensor(1.0), atol=1e-5)


def test_iclsel_controls():
    s, d, e = stub('hyb', n=10, b=2)
    outs = {}
    for mode in ('iclselavg1', 'iclselfix1_all', 'iclselfix1_recent4', 'iclselfix1_phase', 'iclselfix1_topk8', 'iclsel1'):
        s.win_ptr = 0
        if mode == 'iclsel1':
            s.sel_wdump = []
        outs[mode] = _run(s, d, e, mode) - d
    avg = sum(outs[f'iclselfix1_{k}'] for k in R1TeflICL.SEL_SETS) / 4
    assert torch.allclose(outs['iclselavg1'], avg, atol=1e-5)                                        # uniform mix of the fixed sets
    full = (_run(s, d, e, None) - d)
    assert torch.allclose(outs['iclselfix1_all'], full, atol=1e-5)                                   # one GD step on all = hyb
    w = s.sel_wdump[0]
    assert w.shape == (4, 2, 2) and torch.allclose(w.sum(0), torch.ones(2, 2), atol=1e-5)
    mix = sum(w[i][..., None, None] * outs[f'iclselfix1_{k}'] for i, k in enumerate(R1TeflICL.SEL_SETS))
    assert torch.allclose(outs['iclsel1'], mix, atol=1e-5)                                          # validated mix of the same fits


def test_iclselw_given_weights(tmp_path, monkeypatch):
    s, d, e = stub('hyb', n=10, b=2)
    W = torch.softmax(torch.randn(4, 5, 2, generator=torch.Generator().manual_seed(3)), 0)       # (S, N = 5 windows, C)
    torch.save(W, tmp_path / 'w_x.pt')
    monkeypatch.setenv('R1_ICL_SELW', str(tmp_path / 'w'))
    s.win_ptr = 3                                                                                   # windows 3, 4
    out = _run(s, d, e, 'iclselw_x') - d
    fx = []
    for k in R1TeflICL.SEL_SETS:
        s.win_ptr = 0
        fx.append(_run(s, d, e, f'iclselfix1_{k}') - d)
    ref = sum(W[i, 3:5][:, :, None, None] * fx[i] for i in range(4))
    assert torch.allclose(out, ref, atol=1e-5)


def test_hybatt_softmax_lag_prior(monkeypatch):
    import math
    s, d, e = stub('hybatt', n=10, b=2)                                                              # K = 3, lags 12 .. 3
    s.K_last = 3
    s.b_lag = nn.Parameter(torch.zeros(16)); s.log_tau = nn.Parameter(torch.tensor(math.log(1e6)))
    out = R1TeflICL.extra_delta(s, d, e, None)
    k = torch.relu(s.W1(s.demo['k'])); r = s.demo['v'] - s.W2(k)
    ref = r.mean(1)                                                                                  # flat logits -> uniform mean
    assert torch.allclose((out - d).reshape(2, 2, 12), ref, atol=1e-4)
    with torch.no_grad():
        s.b_lag[6] = 50.0                                                                            # only lag 6 = demo j = 12 - 6 = 6
    out = R1TeflICL.extra_delta(s, d, e, None)
    assert torch.allclose((out - d).reshape(2, 2, 12), r[:, 6], atol=1e-4)
    out.sum().backward()
    assert s.b_lag.grad is not None and torch.isfinite(s.b_lag.grad).all() and s.log_tau.grad is not None


def test_two_layer_matches_autograd_step():
    s, d, e = stub('hyb', n=6, b=2)
    d0 = s.W2(torch.relu(s.W1(e))).reshape(d.shape).detach()                                        # TEFL delta, as the model passes it
    s.win_ptr = 0
    hyb = _run(s, d0, e, None)
    s.win_ptr = 0
    assert torch.allclose(_run(s, d0, e, 'icl2l_0'), hyb, atol=1e-5)                                # a1 = 0 == hyb
    a1 = 0.7
    out1 = _run(s, d0, e, 'icl2lw1_0.7').reshape(2, 2, 12)
    for bi in range(2):
        for ci in range(2):
            K = s.demo['k'][bi, :, ci]; V = s.demo['v'][bi, :, ci]
            W1 = s.W1.weight.detach().clone().requires_grad_(True)
            loss = 0.5 * (V - torch.relu(K @ W1.T) @ s.W2.weight.detach().T).square().sum()
            gW, = torch.autograd.grad(loss, W1)
            W1n = W1.detach() - a1 * gW / K.square().sum()                                           # normalised GD step
            ref = torch.relu(e[bi, ci] @ W1n.T) @ s.W2.weight.detach().T
            assert torch.allclose(out1[bi, ci], ref, atol=1e-4), (bi, ci)


def test_z2_normalisation():
    s, d, e = stub('hyb', n=6, b=2)
    d0 = s.W2(torch.relu(s.W1(e))).reshape(d.shape).detach()
    s.win_ptr = 0
    out = _run(s, d0, e, 'icl2lz_0_1').reshape(2, 2, 12)
    k = torch.relu(s.W1(s.demo['k'])); q = torch.relu(s.W1(e)); r = s.demo['v'] - s.W2(k)
    for bi in range(2):
        for ci in range(2):
            K, R = k[bi, :, ci], r[bi, :, ci]
            W2n = s.W2.weight + (R.T @ K) / K.square().sum()                                            # eta = 1: NLMS step on W2
            assert torch.allclose(out[bi, ci], W2n @ q[bi, ci], atol=1e-4)
    s2, d2, e2 = stub('hybz', n=6, b=2)
    s2.W1, s2.W2, s2.demo = s.W1, s.W2, s.demo
    s.win_ptr = 0
    assert torch.allclose(R1TeflICL.extra_delta(s2, d0, e, None), _run(s, d0, e, 'icl2lz_0_1'), atol=1e-5)   # hybz == icl2lz a1 = 0


def test_lb_errors_and_pad():
    """R1_ICL_LB: each origin's forecast sees only its last lb steps; errors normalised by the given stdev; lb_pad + demos puts the query
    at origin L-K*P and every demo value fully inside the window."""
    b, L, c, P, K, lb = 2, 40, 3, 4, 2, 12
    x = torch.randn(b, L, c, generator=torch.Generator().manual_seed(0))
    seen = []
    def fc(xw, k, si):                                                                   # naive: repeat the window mean, record widths
        seen.append(xw.shape[1])
        assert all(bool((x[s_] == xw[i, -1]).all(-1).any()) for i, s_ in enumerate(si.tolist()))   # window i comes from sample si[i]
        return xw.mean(1, keepdim=True).expand(-1, k * P, -1)
    sd = torch.rand(b, 1, c) + 0.5
    E = lb_errors(x, lb, K, P, fc, sd, chunk=5)
    o = list(range(lb, L - K * P + 1, P))
    assert E.shape == (b, len(o), K, c, P) and set(seen) == {lb}
    for m, oo in enumerate(o):
        ref = (x[:, oo:oo + K * P] - x[:, oo - lb:oo].mean(1, keepdim=True)) / sd   # (b, K*P, c)
        assert torch.allclose(E[:, m], ref.reshape(b, K, P, c).permute(0, 1, 3, 2), atol=1e-6)
    Ef, avail = lb_pad(E, K)
    eps, D = demos(Ef, avail, K)
    M = len(o)
    assert torch.allclose(eps, E[:, M - 1].permute(0, 2, 1, 3).reshape(b, c, K * P))  # query = origin L - K*P
    assert D['k'].shape[1] == M - K and bool(D['mask'].all())
    assert torch.allclose(D['v'][:, -1], eps)                                            # latest demo value = the query's own row
    assert torch.allclose(D['k'][:, 0], E[:, 0].permute(0, 2, 1, 3).reshape(b, c, K * P))
