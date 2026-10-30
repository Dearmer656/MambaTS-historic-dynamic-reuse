import torch

import patchmem.eval_freshcast as fc

H, LQ = fc.H, fc.LQ


def setup(seed, T=3600, C=5):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(T, C, generator=g).cumsum(0) * 0.1 + torch.randn(T, C, generator=g)
    return X


def parts(X, t, c):
    ws = fc.win_stats(X)
    sd = fc.series_desc(X[:2000])
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
    sp = fc.build(X, ws, sd, order, t, c)
    R = fc.readout(X, ws, t, c, sp['s'], sp['p'], sp['val'])
    return ws, sp, R


def test_causal():
    """Changing y_{>=t} leaves every candidate, descriptor, readout and the RKR output unchanged."""
    X = setup(1)
    t, c = torch.tensor([3300, 3300, 3400]), torch.tensor([0, 3, 1])
    torch.manual_seed(0)
    model = fc.RKR()
    _, a, Ra = parts(X, t, c)
    X2 = X.clone()
    X2[3300:] = 9.0
    _, b, Rb = parts(X2, t, c)
    for k in ('s', 'p', 'val', 'valid'):
        assert torch.equal(a[k][:2], b[k][:2]), k
    assert torch.allclose(a['z'][:2], b['z'][:2]) and torch.allclose(a['u'][:2], b['u'][:2])
    assert torch.allclose(Ra[:2], Rb[:2])
    ra = (model(a['z'], a['u'], a['valid'])[..., None] * Ra).sum(1)
    rb = (model(b['z'], b['u'], b['valid'])[..., None] * Rb).sum(1)
    assert torch.allclose(ra[:2], rb[:2], atol=1e-6)
    assert bool((a['s'][0] + torch.where(a['p'][0] > 0, a['p'][0], H) <= 3300).all())


def test_explicit():
    """One query by hand: seasonal tiling, readouts, shape-analog top-N by cosine over recent + distant pools."""
    X = setup(2)
    t0, c0 = 3500, 2
    t, c = torch.tensor([t0]), torch.tensor([c0])
    _, sp, R = parts(X, t, c)
    Xd = X.double()

    def st(s):
        w = Xd[s - LQ:s, c0]
        return w.mean(), w.std(unbiased=False)
    mq, sq = st(t0)
    q = Xd[t0 - LQ:t0, c0] - mq
    for i, (age, p) in enumerate(fc.SEAS):
        s = t0 - age
        cont = torch.stack([Xd[s + h % p, c0] for h in range(H)])
        mc, sc = st(s)
        assert torch.allclose(R[0, i].double(), cont, atol=1e-4)
        assert torch.allclose(R[0, fc.N_SEAS + i].double(), mq + sq / sc * (cont - mc), atol=1e-3)
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
    dist = [int(s) for s in order if LQ <= s < t0 - fc.K_STEPS][:fc.Q_DIST]
    pool = list(range(t0 - fc.K_STEPS, t0 - H + 1)) + dist
    cos = []
    for s in pool:
        mc, sc = st(s)
        cos.append(float((Xd[s - LQ:s, c0] - mc) @ q / (LQ * sc * sq)))
    top = sorted(range(len(pool)), key=lambda i: -cos[i])[:fc.N_SHAPE]
    want = {pool[i] for i in top}
    got = set(sp['s'][0, 2 * fc.N_SEAS:].tolist())
    assert len(want ^ got) <= 2, len(want ^ got)                        # ties at the boundary only
    s = int(sp['s'][0, 2 * fc.N_SEAS])
    mc, sc = st(s)
    assert torch.allclose(R[0, 2 * fc.N_SEAS].double(), mq + sq / sc * (Xd[s:s + H, c0] - mc), atol=1e-3)
    assert sp['z'].shape[-1] == fc.NZ and sp['u'].shape[-1] == fc.NU


def test_untrained_grid():
    """The closed-form grid SSE equals the direct SSE of the chosen per-channel rule."""
    X = setup(3)
    ws = fc.win_stats(X)
    sd = fc.series_desc(X[:2000])
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
    origins = torch.tensor([3200, 3250, 3300])
    eps, ur = fc.untrained_fit(X, ws, sd, order, origins)
    C = X.shape[1]
    sse = torch.zeros(C, dtype=torch.float64)
    for t0 in origins.tolist():
        c = torch.arange(C)
        t = torch.full_like(c, t0)
        sp = fc.build(X, ws, sd, order, t, c)
        R = fc.readout(X, ws, t, c, sp['s'], sp['p'], sp['val'])
        r = fc.untrained_r(R, ur, c)
        y = X[t[:, None] + torch.arange(H), c[:, None]]
        sse += (r - y).double().square().sum(1)
    assert torch.allclose(eps.double(), sse / (len(origins) * H), rtol=1e-4)
