import torch

import patchmem.eval_obs_icl as oi

H, R = oi.H, oi.R


def setup(seed, T=3600, C=5):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(T, C, generator=g).cumsum(0) * 0.1 + torch.randn(T, C, generator=g)
    return X, g


def test_causal():
    """Changing y_{>=t} leaves mu_t, sigma_t and every v^ unchanged (demo continuations end by t)."""
    X, _ = setup(1)
    t, c = torch.tensor([3200, 3200]), torch.tensor([0, 3])
    ages = torch.as_tensor(oi.pool_ages('daily'))
    for P in oi.PS:
        a = oi.predict(X, oi.rolling(X, P), t, c, ages)
        X2 = X.clone()
        X2[3200:] = 9.0
        b = oi.predict(X2, oi.rolling(X2, P), t, c, ages)
        assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
        for k in a[2]:
            assert torch.allclose(a[2][k], b[2][k], atol=1e-5), k


def test_explicit():
    """One case by hand: normalization, B softmax, C1 dot-product sum, C2 least squares."""
    X, _ = setup(2)
    t, c, P = 3300, 2, 168
    ages = oi.pool_ages('phase')
    Xd = X.double()

    def ms(i):
        w = Xd[i - P:i, c]
        return w.mean(), w.std(unbiased=False).clamp_min(oi.SIG_FLOOR)
    Ks, Vs = [], []
    for a in ages:
        s = t - a
        m, sd = ms(s)
        Ks.append((Xd[s - R:s, c] - m) / sd)
        Vs.append((Xd[s:s + H, c] - m) / sd)
    K, V = torch.stack(Ks), torch.stack(Vs)
    m, sd = ms(t)
    kt = (Xd[t - R:t, c] - m) / sd
    dot = K @ kt / R
    N = len(ages)
    want = {('B', 2.0): torch.softmax(2.0 * dot, 0) @ V, ('C1', 0.0): dot @ V / N}
    KK = K.T @ K / N
    lam = 0.1
    want['C2', lam] = kt @ torch.linalg.solve(KK + lam * KK.diagonal().mean() * torch.eye(R, dtype=torch.float64),
                                              K.T @ V / N)
    mu_t, sd_t, got = oi.predict(X, oi.rolling(X, P), torch.tensor([t]), torch.tensor([c]), torch.as_tensor(ages))
    assert abs(float(mu_t) - float(m)) < 1e-4 and abs(float(sd_t) - float(sd)) < 1e-4
    for k, v in want.items():
        assert torch.allclose(got[k][0].double(), v, rtol=1e-3, atol=1e-3), k


def test_mse_stats():
    """mse() from accumulated second moments equals the direct MSE of y^ = f + b (mu + eta sigma v^ - f)."""
    X, g = setup(3)
    t = torch.tensor([3200, 3250, 3300, 3350])
    c = torch.tensor([0, 1, 2, 4])
    e = torch.randn(4, H, generator=g)
    st = {24: oi.rolling(X, 24)}
    ages = oi.pool_ages('phase')
    acc, n = oi.accumulate(X, st, (t, c, e), ages, 3)
    eta, b = 0.7, torch.full((H,), 0.4, dtype=torch.float64)
    mu_t, sd_t, vh = oi.predict(X, st[24], t, c, torch.as_tensor(ages))
    y = X[t[:, None] + torch.arange(H), c[:, None]].double()
    f = y - e.double()
    yh = f + b * (mu_t.double() + eta * sd_t.double() * vh['B', 4.0].double() - f)
    want = float((y - yh).square().mean())
    got = oi.mse(acc[24, 'B', 4.0], eta, b, n)
    assert abs(got - want) < 1e-6 * max(1.0, want), (got, want)


def test_flat_channel():
    """A constant channel (all normalized keys 0) must not make C2 singular."""
    X, _ = setup(4)
    X[:, 1] = 3.0
    _, _, got = oi.predict(X, oi.rolling(X, 24), torch.tensor([3200]), torch.tensor([1]),
                           torch.as_tensor(oi.pool_ages('daily')))
    for k, v in got.items():
        assert torch.isfinite(v).all(), k
