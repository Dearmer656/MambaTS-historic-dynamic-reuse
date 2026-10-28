import torch

import patchmem.eval_error_path as ep

L, H = ep.L, ep.H
AGES = [192, 200, 230, 264]


def synth(seed=0, T=1200, C=3):
    g = torch.Generator().manual_seed(seed)
    origins = torch.arange(L, T - H + 1)
    return dict(E=torch.randn(len(origins), H, C, generator=g), X=torch.randn(T, C, generator=g),
                M=torch.randn(T, 4, generator=g), origins=origins)


def as_double(b):
    return {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in b.items()}


def models(beta_mode):
    torch.manual_seed(0)
    return {v: ep.ErrorPath(v, 4, beta_mode).double() for v in ep.VARIANTS}


def run_all(ms, data, t, c):
    b = as_double(ep.gather(data, t, c, AGES, True, True))
    for v in ep.STATE_VARIANTS:
        ms[v].pgen.manual_seed(0)                                        # same Sperm draw in every call
    return {v: m(b)[0] for v, m in ms.items()}


def test_causality_unknown_future_is_ignored():
    for mode in ('two', 'learn'):
        ms = models(mode)
        data = synth()
        t, c = torch.tensor([900]), torch.tensor([2])
        ref = run_all(ms, data, t, c)
        pert = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
        tq = int(t[0])
        o = torch.arange(pert['E'].shape[0])[:, None] + L                  # origin of each row
        unknown = (o + torch.arange(H)[None]) >= tq                        # y_{o+h} not observed before t
        pert['E'][unknown] += 100 * torch.randn(int(unknown.sum()), pert['E'].shape[2])
        pert['X'][tq:] += 100 * torch.randn_like(pert['X'][tq:])
        got = run_all(ms, pert, t, c)
        for v in ep.VARIANTS:
            assert torch.equal(got[v], ref[v]), (mode, v)
        pert['X'][tq - 1] += 1.0                                           # an observed input does matter
        moved = run_all(ms, pert, t, c)
        for v in ('End', 'C', 'Cmean', 'Crev', 'Spre', 'Sperm', 'Smean'):
            assert not torch.equal(moved[v], ref[v]), (mode, v)


def explicit_P(u, beta, a):
    d = u.shape[-1]
    P = torch.eye(d, dtype=u.dtype)
    for k in range(a):                                     # P = A[k=0] @ A[k=1] @ ... @ A[k=a-1]
        P = P @ (torch.eye(d, dtype=u.dtype) - beta[k] * torch.outer(u[k], u[k]))
    return P


def test_path_products_bruteforce_order_and_norm():
    torch.manual_seed(1)
    n, d, K = len(AGES), ep.D_EMB, max(AGES)
    fq = torch.randn(2, d, dtype=torch.float64); fk = torch.randn(2, n, d, dtype=torch.float64)
    u = torch.nn.functional.normalize(torch.randn(2, K, d, dtype=torch.float64), dim=-1)
    for beta in (2 * torch.rand(2, K, dtype=torch.float64), torch.full((2, K), 2.0, dtype=torch.float64)):
        ms = models('learn')
        sc, _ = ms['C'].path_scores(fq, fk, AGES, u, beta)
        sr, _ = ms['Crev'].path_scores(fq, fk, AGES, u, beta)
        pk = ep.sweep(fk, u, beta, AGES, ep.factor_order(AGES, 'P'))
        ptk = ep.sweep(fk, u, beta, AGES, ep.factor_order(AGES, 'PT'))
        perm = ep.sweep(fk, u, beta, AGES, ep.factor_order(AGES, 'perm', torch.Generator().manual_seed(0)))
        for b in range(2):
            for j, a in enumerate(AGES):
                P = explicit_P(u[b], beta[b], a)
                assert abs(float(fq[b] @ P @ fk[b, j]) - float(sc[b, j])) < 1e-9
                assert abs(float(fq[b] @ P.T @ fk[b, j]) - float(sr[b, j])) < 1e-9
                assert torch.allclose(P @ fk[b, j], pk[b, j]) and torch.allclose(P.T @ fk[b, j], ptk[b, j])
        assert (sc - sr).abs().max() > 1e-3 and ((fq[:, None] * perm).sum(-1) - sc).abs().max() > 1e-3
    # beta = 2: exact reflections keep the norm over the whole path
    assert torch.allclose(pk.norm(dim=-1), fk.norm(dim=-1)) and torch.allclose(perm.norm(dim=-1), fk.norm(dim=-1))


def test_identity_transition_matches_Bp():
    """A_r = I (beta = 0) with C sharing Bp's parameters gives Bp case by case."""
    torch.manual_seed(3)
    bp = ep.ErrorPath('Bp', 4).double()
    c = ep.ErrorPath('C', 4, 'learn').double()
    c.load_state_dict({**c.state_dict(), **bp.state_dict()})
    c.transitions = lambda b: (torch.nn.functional.normalize(torch.randn(*b['yr'].shape[:2], ep.D_EMB,
                                                                         dtype=torch.float64), dim=-1),
                               torch.zeros(b['yr'].shape[:2], dtype=torch.float64))
    data = synth(4)
    b = as_double(ep.gather(data, torch.tensor([880, 930, 1000]), torch.tensor([0, 1, 2]), AGES, True))
    assert torch.allclose(c(b)[0], bp(b)[0], atol=1e-12)


def test_model_order_diag_matches_C_and_A_is_mean():
    ms = models('two')
    data = synth(5)
    b = as_double(ep.gather(data, torch.tensor([950, 1000]), torch.tensor([1, 0]), AGES, True))
    assert torch.allclose(ms['C'](b)[0], ms['C'](b, ep.factor_order(AGES, 'P'))[0], atol=1e-10)
    out, _ = ms['A'](b)
    assert torch.allclose(out, 0.3 * b['V'].mean(1))


def test_state_bruteforce_order_and_norm():
    """Spre state = A_{r-1} ... A_{r-W} h0 (explicit matrices), norm preserved; Sperm draw changes it."""
    ms = models('two')
    m = ms['Spre']
    data = synth(6)
    b = as_double(ep.gather(data, torch.tensor([990]), torch.tensor([1]), AGES, True, True))
    idx = m.state_order(False)
    hq = m.state(b['yqw'], idx)[0]                                       # m d
    u = torch.nn.functional.normalize(m.strans(ep.window_features(b['yqw'][0])), dim=-1)   # W d (k = 0 latest)
    P = torch.eye(ep.D_EMB, dtype=torch.float64)
    for k in range(ep.W_STATE):                                          # P = A_{r-1} @ ... @ A_{r-W}
        P = P @ (torch.eye(ep.D_EMB, dtype=torch.float64) - 2 * torch.outer(u[k], u[k]))
    assert torch.allclose(hq, m.h0.double() @ P.T, atol=1e-10)
    assert torch.allclose(hq.norm(dim=-1), m.h0.double().norm(dim=-1))
    # demo state for demo j matches the query-style state of origin s_j
    j = 2
    bs = as_double(ep.gather(data, torch.tensor([990 - AGES[j]]), torch.tensor([1]), AGES, True, True))
    assert torch.allclose(m.state(b['ysw'], idx)[0, j], m.state(bs['yqw'], idx)[0])
    perm = torch.randperm(ep.W_STATE, generator=torch.Generator().manual_seed(1))
    assert (m.state(b['yqw'], perm)[0] - hq).abs().max() > 1e-3


def test_smean_is_order_free():
    ms = models('two')
    m = ms['Smean']
    b = as_double(ep.gather(synth(7), torch.tensor([990]), torch.tensor([0]), AGES, True, True))
    perm = torch.randperm(ep.W_STATE, generator=torch.Generator().manual_seed(2))
    yw = b['ysw']
    assert torch.allclose(m.state(yw, perm), m.state(yw[..., perm, :], perm), atol=1e-12)
    assert torch.allclose(m.state(yw, perm).norm(dim=-1), m.gain.double().abs().expand(1, len(AGES), -1))
