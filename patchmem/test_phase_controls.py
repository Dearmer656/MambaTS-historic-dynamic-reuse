import torch
from patchmem.eval_itransformer_tefl_icl import score_grid
from patchmem.eval_itransformer_tefl_phase_controls import collect, settings_mse, derangement

L, H = 96, 192


def setup():
    torch.manual_seed(0)
    E = torch.randn(900, H, 3)
    W1 = torch.randn(16, H, dtype=torch.float64) / H ** 0.5
    W2 = torch.randn(H, 16, dtype=torch.float64) / 4
    history = 768
    origins = torch.arange(history, 880)
    ages = [192, 240, 288, 336, 384, 432, 480]
    return E, W1, W2, history, origins, ages


def test_icl_w2_matches_score_grid():
    E, W1, W2, N, origins, ages = setup()
    cands = [(0.0, a) for a in (0.1, 0.3, 1.0, 3.0)]
    ref, _ = score_grid(E, origins, W1, W2, N, 1, 5, cands, ages)
    S0, S1, S2, _ = collect(E, origins, W1, W2, ages, 5, 2, 0)
    mse = settings_mse(S0, S1, S2)
    for _, a in cands:
        assert torch.allclose(mse[('icl_w2', a)], ref[0.0, a], atol=1e-9)


def test_phase_mean_direct():
    E, W1, W2, N, origins, ages = setup()
    S0, S1, S2, _ = collect(E, origins, W1, W2, ages, 7, 1, 0)
    mse = settings_mse(S0, S1, S2)[('phase_mean', 0.5)]
    t = origins[3]
    q = E[t - H - L].T.double(); tg = E[t - L].T.double()
    s = t - torch.tensor(ages)
    K = E[s - H - L].permute(2, 0, 1).double(); V = E[s - L].permute(2, 0, 1).double()
    r = V - torch.relu(K @ W1.T) @ W2.T
    pred = torch.relu(q @ W1.T) @ W2.T + 0.5 * r.mean(1)
    assert abs(float((tg - pred).square().mean()) - float(mse[3])) < 1e-9


def test_derangement():
    pi = derangement((4, 5), 7, torch.Generator().manual_seed(1), 'cpu')
    assert (pi.sort(-1).values == torch.arange(7)).all()
    assert not (pi == torch.arange(7)).any()


def test_avg_icl_consistency():
    import patchmem.eval_itransformer_tefl_avg_icl as m
    from patchmem.eval_itransformer_tefl_phase_controls import settings_mse
    E, W1, W2, N, origins, ages = setup()
    S0, S1, S2, _ = collect(E, origins, W1, W2, ages, 5, 1, 0)
    ref = settings_mse(S0, S1, S2)[('phase_mean', 1.0)]
    T0, T1, T2, D = m.collect(E, origins, W1, W2, ages, 5)
    assert torch.allclose(m.mse(T0, T1, T2, 'avg', 1.0), ref, atol=1e-9)
    # Very large temperature -> uniform softmax -> centered values cancel -> same as avg.
    old = m.TAU_COS
    m.TAU_COS = 1e12
    try:
        U0, U1, U2, _ = m.collect(E, origins, W1, W2, ages, 5)
        assert torch.allclose(m.mse(U0, U1, U2, 'icl_cos', 1.0), ref, atol=1e-9)
    finally:
        m.TAU_COS = old
    assert m.phase_ages(168, 3) == [336, 504, 672] and m.phase_ages(24, 2) == [192, 216]
    assert m.phase_ages(1008, 2) == [1008, 2016]
