import torch

import patchmem.eval_error_path as ep

L, H = ep.L, ep.H


def synth(seed=0, T=1200, C=3):
    g = torch.Generator().manual_seed(seed)
    origins = torch.arange(L, T - H + 1)
    return dict(E=torch.randn(len(origins), H, C, generator=g), X=torch.randn(T, C, generator=g),
                M=torch.randn(T, 4, generator=g), origins=origins)


AGES = [192, 200, 230, 264]


def models():
    torch.manual_seed(0)
    return {v: ep.ErrorPath(v, 4).double() for v in ep.VARIANTS}


def run_all(data, t, c):
    b = ep.gather(data, t, c, AGES, True)
    b = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in b.items()}
    return {v: m(b)[0] for v, m in MODELS.items()}


MODELS = models()


def test_causality_unknown_future_is_ignored():
    data = synth()
    t = torch.tensor([900]); c = torch.tensor([2])
    ref = run_all(data, t, c)
    pert = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
    tq = int(t.min())
    o = torch.arange(pert['E'].shape[0])[:, None] + L                  # origin of each row
    unknown = (o + torch.arange(H)[None]) >= tq                        # y_{o+h} not yet observed at tq
    pert['E'][unknown] += 100 * torch.randn(int(unknown.sum()), pert['E'].shape[2])
    pert['X'][tq:] += 100 * torch.randn_like(pert['X'][tq:])
    got = run_all(pert, t, c)
    for v in ep.VARIANTS:
        assert torch.equal(got[v], ref[v]), v
    # Sanity: perturbing one observed input (y at t-1) does change the path variants.
    pert['X'][tq - 1] += 1.0
    moved = run_all(pert, t, c)
    for v in ('End', 'C', 'Crev'):
        assert not torch.equal(moved[v], ref[v]), v


def test_path_scores_bruteforce_and_order():
    torch.manual_seed(1)
    n, d, K = len(AGES), ep.D_EMB, max(AGES)
    fq = torch.randn(2, d, dtype=torch.float64); fk = torch.randn(2, n, d, dtype=torch.float64)
    u = torch.nn.functional.normalize(torch.randn(2, K, d, dtype=torch.float64), dim=-1)
    beta = 2 * torch.rand(2, K, dtype=torch.float64)
    ages = torch.tensor(AGES)
    sc, _ = MODELS['C'].path_scores(fq, fk, ages, u, beta)
    sr, _ = MODELS['Crev'].path_scores(fq, fk, ages, u, beta)
    for b in range(2):
        A = [torch.eye(d, dtype=torch.float64) - beta[b, k] * torch.outer(u[b, k], u[b, k]) for k in range(K)]
        for j, a in enumerate(AGES):
            P = torch.eye(d, dtype=torch.float64)          # P_j = A_{t-1} ... A_{s_j} = A[k=0] @ ... @ A[k=a-1]
            for k in range(a):
                P = P @ A[k]
            assert abs(float(fq[b] @ P @ fk[b, j]) - float(sc[b, j])) < 1e-9
            assert abs(float(fq[b] @ P.T @ fk[b, j]) - float(sr[b, j])) < 1e-9
    assert (sc - sr).abs().max() > 1e-3                    # order matters


def test_zero_beta_reduces_to_dot():
    n, d, K = len(AGES), ep.D_EMB, max(AGES)
    fq = torch.randn(3, d, dtype=torch.float64); fk = torch.randn(3, n, d, dtype=torch.float64)
    u = torch.nn.functional.normalize(torch.randn(3, K, d, dtype=torch.float64), dim=-1)
    beta = torch.zeros(3, K, dtype=torch.float64)
    for v in ('C', 'Crev'):
        s, dev = MODELS[v].path_scores(fq, fk, torch.tensor(AGES), u, beta)
        assert torch.allclose(s, (fq[:, None] * fk).sum(-1)) and float(dev.abs().max()) < 1e-12


def test_variant_A_is_scaled_mean():
    data = synth(2)
    b = ep.gather(data, torch.tensor([1000]), torch.tensor([1]), AGES, False)
    out, _ = ep.ErrorPath('A', 4)(b)
    assert torch.allclose(out, 0.3 * b['V'].mean(1), atol=1e-6)
