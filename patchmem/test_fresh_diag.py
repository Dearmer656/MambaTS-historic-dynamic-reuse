import torch

import patchmem.eval_fresh_diag as fd
from patchmem.test_error_path import synth

AGES = [192, 200, 230, 264]


def setup(seed, C=20):
    data = synth(seed, T=1700, C=C)
    data['origins'] = torch.tensor([1200])
    P = torch.linalg.qr(torch.randn(C, fd.R, generator=torch.Generator().manual_seed(seed)))[0]
    return data, P


def test_causal():
    """Changing every error entry not yet observed at origin t (o + h >= t) and y_{>=t} only moves target terms."""
    data, P = setup(21)
    t = int(data['origins'][0])
    a = fd.stats(data, AGES, P, 'cpu', 64)
    E = data['E'].clone()
    o = torch.arange(len(E))[:, None] + fd.L
    hidden = (o + torch.arange(fd.H)[None]) >= t                       # rows x H
    E[hidden] = torch.randn(int(hidden.sum()), E.shape[2], generator=torch.Generator().manual_seed(5))
    X = data['X'].clone()
    X[t:] = 7.0
    b = fd.stats(dict(data, E=E, X=X), AGES, P, 'cpu', 64)
    for k in ('G', 'Sr', 'rr'):
        assert torch.allclose(a[k], b[k]), k
    assert not torch.allclose(a['TT'], b['TT'])


def test_features_and_ridge():
    """Explicit features of one case: Gram matches, and the Gram-based ridge SSE equals a direct least squares."""
    data, P = setup(22, C=20)
    t = int(data['origins'][0])
    E, X, L, H = data['E'].double(), data['X'].double(), fd.L, fd.H
    st = fd.stats(data, AGES, P, 'cpu', 64)
    rows, tgt, rets = [], [], []
    for c in range(E.shape[2]):
        f1 = [E[t - j - L, 0, c] for j in range(1, fd.NJ + 1)]
        f2 = [E[t - k - L, k - 1, c] for k in range(2, fd.NK + 1)]
        z = [E[t - j - L, 0, :] @ P.double() for j in range(1, fd.NX + 1)]
        rows.append(torch.cat([torch.ones(1), torch.stack(f1 + f2), torch.cat(z)]))
        cs = torch.nn.functional.cosine_similarity
        lg = torch.stack([cs(E[t - H - L, :, c], E[t - a - H - L, :, c], dim=0)
                          + 4 * cs(X[t - 96:t, c], X[t - a - 96:t - a, c], dim=0) for a in AGES])
        w = torch.softmax(lg, 0)
        rets.append(sum(w[i] * E[t - a - L, :, c] for i, a in enumerate(AGES)))
        tgt.append(E[t - L, :, c])
    S, T, Rt = torch.stack(rows), torch.stack(tgt), torch.stack(rets)
    assert torch.allclose(st['G'], S.T @ S, rtol=1e-4, atol=1e-4)
    assert torch.allclose(st['Sr'], S.T @ Rt, rtol=1e-4, atol=1e-4)
    idx, use_ret = fd.arm_index('fresh+ret')
    W = fd.fit(st, idx, use_ret, 1e-3)
    got = fd.sse(st, idx, use_ret, W)
    for h in (0, 50, H - 1):
        Xh = torch.cat([S[:, idx], Rt[:, h:h + 1]], 1)
        want = float((T[:, h] - Xh @ W[h]).square().sum())
        assert abs(float(got[h]) - want) < 1e-4 * max(1.0, want), (h, float(got[h]), want)
