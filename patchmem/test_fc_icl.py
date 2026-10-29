import torch

import patchmem.eval_fc_icl as fi

H, NB = fi.H, fi.NB


def test_weights_direct():
    """icl_weights from prefix sums equals the direct demo sums over origins k-H-N+1 .. k-H (clipped at 0)."""
    g = torch.Generator().manual_seed(0)
    n, C = 900, 3
    e = torch.randn(n, C, H, generator=g, dtype=torch.float64)
    d = 0.5 * e + torch.randn(n, C, H, generator=g, dtype=torch.float64)
    bk = fi.buckets()
    pb = lambda x: torch.zeros(NB, C, dtype=torch.float64).index_add_(0, bk, x.T)
    P = torch.zeros(n + 1, 2, NB, C, dtype=torch.float64)
    for k in range(n):
        P[k + 1, 0] = P[k, 0] + pb(e[k] * d[k])
        P[k + 1, 1] = P[k, 1] + pb(d[k].square())
    D0 = torch.rand(NB, C, generator=g, dtype=torch.float64) + 0.5
    w_hat = 0.4
    for k in (0, H - 1, H, H + 5, 800):
        W = fi.icl_weights(P, k, (24, 168), (0.0, 4.0), w_hat, D0)
        for N in (24, 168):
            idx = [s for s in range(k - H - N + 1, k - H + 1) if s >= 0]
            for kap in (0.0, 4.0):
                if idx:
                    ed = torch.stack([pb(e[s] * d[s]) for s in idx]).sum(0)
                    dd = torch.stack([pb(d[s].square()) for s in idx]).sum(0)
                else:
                    ed = dd = torch.zeros(NB, C, dtype=torch.float64)
                for form in ('scalar', 'lead'):
                    a, b, p0 = (ed.sum(0, keepdim=True), dd.sum(0, keepdim=True), D0.sum(0, keepdim=True)) \
                        if form == 'scalar' else (ed, dd, D0)
                    num, den = a + kap * w_hat * p0, b + kap * p0
                    want = torch.where(den > 0, num / den.clamp_min(1e-12), torch.full_like(den, w_hat)).clamp(0, 1)
                    assert torch.allclose(W[form, N, kap], want.expand(NB, -1)), (k, N, kap, form)
    # demo set never includes origin indices > k - H
    P2 = P.clone()
    P2[n - H + 1:] += 7.0
    assert all(torch.equal(a, b) for a, b in zip(fi.icl_weights(P, n - 1, (24,), (1.0,), w_hat, D0).values(),
                                                  fi.icl_weights(P2, n - 1, (24,), (1.0,), w_hat, D0).values()))
