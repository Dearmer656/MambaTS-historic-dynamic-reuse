import torch

import patchmem.eval_obs_icl as oi

H, R = oi.H, oi.R


def setup(seed, T=3600, C=5):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(T, C, generator=g).cumsum(0) * 0.1 + torch.randn(T, C, generator=g)
    return X, g


def cands(pool):
    a, p = oi.pool_cands(pool)
    return torch.as_tensor(a), torch.as_tensor(p)


def test_causal():
    """Changing y_{>=t} leaves mu_t, sigma_t and every v^ unchanged, for age pools and the tiled pool."""
    X, _ = setup(1)
    t, c = torch.tensor([3200, 3200]), torch.tensor([0, 3])
    for pool in ('daily', 'tiled'):
        ages, per = cands(pool)
        for P in oi.PS:
            a = oi.predict(X, oi.rolling(X, P), t, c, ages, per)
            X2 = X.clone()
            X2[3200:] = 9.0
            b = oi.predict(X2, oi.rolling(X2, P), t, c, ages, per)
            assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
            for k in a[2]:
                assert torch.allclose(a[2][k], b[2][k], atol=1e-5), (pool, k)


def test_explicit():
    """One case by hand on the tiled pool: tiling, both readouts, naive, B softmax, C1, C2."""
    X, _ = setup(2)
    t, c, P = 3300, 2, 168
    ages, per = oi.pool_cands('tiled')
    Xd = X.double()

    def ms(i):
        w = Xd[i - P:i, c]
        return w.mean(), w.std(unbiased=False).clamp_min(oi.SIG_FLOOR)
    m, sd = ms(t)
    Ks, Vs, Vv = [], [], []
    for a, p in zip(ages, per):
        s = t - a
        ms_, ss_ = ms(s)
        cont = torch.stack([Xd[s + h % p, c] for h in range(H)])
        Ks.append((Xd[s - R:s, c] - ms_) / ss_)
        Vs.append((cont - ms_) / ss_)
        Vv.append((cont - m) / sd)
    K, V, VV = torch.stack(Ks), torch.stack(Vs), torch.stack(Vv)
    kt = (Xd[t - R:t, c] - m) / sd
    dot = K @ kt / R
    N = len(ages)
    want = {('B', 'shape', 2.0): torch.softmax(2.0 * dot, 0) @ V, ('C1', 'value', 0.0): dot @ VV / N,
            ('naive24', 'value', 0.0): (torch.stack([Xd[t - 24 + h % 24, c] for h in range(H)]) - m) / sd,
            ('naive168', 'value', 0.0): (torch.stack([Xd[t - 168 + h % 168, c] for h in range(H)]) - m) / sd}
    K2, V2, d2 = torch.cat([K, K]), torch.cat([V, VV]), torch.cat([dot, dot])
    want['B', 'both', 4.0] = torch.softmax(4.0 * d2, 0) @ V2
    KK = K.T @ K / N
    lam = 0.1
    want['C2', 'shape', lam] = kt @ torch.linalg.solve(KK + (lam * KK.diagonal().mean() + 1e-6)
                                                       * torch.eye(R, dtype=torch.float64), K.T @ V / N)
    mu_t, sd_t, got = oi.predict(X, oi.rolling(X, P), torch.tensor([t]), torch.tensor([c]), torch.as_tensor(ages),
                                 torch.as_tensor(per))
    assert abs(float(mu_t) - float(m)) < 1e-4 and abs(float(sd_t) - float(sd)) < 1e-4
    for k, v in want.items():
        assert torch.allclose(got[k][0].double(), v, rtol=1e-3, atol=1e-3), k


def test_mse_stats():
    """mse_ac from accumulated moments equals the direct MSE of y^ = f + a (mu - f) + c sigma v^; fit_ac is LS."""
    X, g = setup(3)
    t = torch.tensor([3200, 3250, 3300, 3350])
    c = torch.tensor([0, 1, 2, 4])
    e = torch.randn(4, H, generator=g)
    st = {24: oi.rolling(X, 24)}
    ages, per = oi.pool_cands('phase')
    acc, n = oi.accumulate(X, st, (t, c, e), ages, per, 3)
    key = (24, 'B', 'value', 4.0)
    mu_t, sd_t, vh = oi.predict(X, st[24], t, c, torch.as_tensor(ages), torch.as_tensor(per))
    y = X[t[:, None] + torch.arange(H), c[:, None]].double()
    f = y - e.double()
    g1, g2 = mu_t.double() - f, sd_t.double() * vh[key[1:]].double()
    wa, wc = torch.full((H,), 0.3, dtype=torch.float64), torch.full((H,), -0.2, dtype=torch.float64)
    want = float((y - (f + wa * g1 + wc * g2)).square().mean())
    assert abs(oi.mse_ac(acc[key], wa, wc, n) - want) < 1e-6 * max(1.0, want)
    for pl in (False, True):
        fa, fc = oi.fit_ac(acc[key], pl)
        if pl:
            for h in (0, H - 1):
                w = torch.linalg.lstsq(torch.stack([g1[:, h], g2[:, h]], 1), e.double()[:, h:h + 1]).solution[:, 0]
                assert torch.allclose(torch.stack([fa[h], fc[h]]), w, atol=1e-5), h
        else:
            w = torch.linalg.lstsq(torch.stack([g1.flatten(), g2.flatten()], 1),
                                   e.double().flatten()[:, None]).solution[:, 0]
            assert torch.allclose(torch.stack([fa[0], fc[0]]), w, atol=1e-5)


def test_flat_channel():
    """A constant channel (all normalized keys 0) must not make C2 singular."""
    X, _ = setup(4)
    X[:, 1] = 3.0
    ages, per = cands('tiled')
    _, _, got = oi.predict(X, oi.rolling(X, 24), torch.tensor([3200]), torch.tensor([1]), ages, per)
    for k, v in got.items():
        assert torch.isfinite(v).all(), k
