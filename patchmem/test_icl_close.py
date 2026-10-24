"""Tests for patchmem/icl_close.py (CPU, synthetic, grid stride 48 so H = 7 grid steps)."""
import numpy as np
import torch
import patchmem.icl_close as ic

ic.dev = 'cpu'
SD = 48
HS_ = ic.H // SD


def synth(N=420, C=2, din=6, mode='bias', seed=0):
    g = torch.Generator().manual_seed(seed)
    own = 0.3 * torch.randn(N, ic.H, C, generator=g)
    eps = torch.randn(N, C, din, generator=g)
    if mode == 'bias':                                                                  # slow common offset: the demo mean predicts it
        off = torch.cumsum(0.05 * torch.randn(N, 1, C, generator=g), 0)
        own = own + off
    elif mode == 'key':                                                                 # value determined by a 1-d direction of eps
        protos = torch.randn(2, ic.H, generator=g)
        lab = (eps[:, :, 0] > 0).long()                                                 # (N, C); separable by a dot product
        own = protos[lab].permute(0, 2, 1) + 0.05 * own
    return {'own': own, 'bb': own.clone(), 'eps': eps}


def test_demo_idx_no_leak():
    q = torch.tensor([200, 300])
    ds = list(range(ic.H, 8 * ic.H + 1, ic.P))
    di = ic.demo_idx(q, ds, SD)
    assert (di <= q[:, None] - HS_).all() and di[0, 0] == 200 - HS_              # every stored future ends by t


def test_part_a_bias():
    D = synth()
    qi = np.arange(ic.EXT // SD, 420)
    rows, fx, ad = ic.eval_a(ic.part_a(D, qi, SD, 336, 'own'), ic.part_a(D, qi, SD, 336, 'own'))
    assert rows['mean2H'][1] > 0.2, rows                                                # the offset is recovered
    assert rows['comb'][0] >= max(v[0] for k, v in rows.items() if k.startswith('mean')) - 1e-9


def test_part_b_ridge_learns_slope():
    """value_s = A key_s (H -> H linear): ridge beats the mean and its shuffle."""
    g = torch.Generator().manual_seed(1)
    N, C = 420, 2
    own = torch.zeros(N, ic.H, C)
    own[:HS_] = torch.randn(HS_, ic.H, C, generator=g)
    rho = -0.8
    for i in range(HS_, N):                                                              # e_s = rho * e_{s-H} + noise
        own[i] = rho * own[i - HS_] + 0.6 * torch.randn(ic.H, C, generator=g)
    D = {'own': own, 'bb': own.clone(), 'eps': torch.zeros(N, C, 3)}
    qi = np.arange(ic.EXT // SD, N)
    st = ic.part_b(D, qi, SD, 8, 'own', torch.Generator().manual_seed(2))
    ev = {c: ic.eval_b(st[c], st[c]) for c in st}
    assert ev[('rdg', 1.0)]['test'] > ev[('mean', 0)]['test'] + 0.2, (ev[('rdg', 1.0)], ev[('mean', 0)])
    assert ev[('rdg', 1.0)]['sh'] < ev[('rdg', 1.0)]['test'] - 0.2


def test_part_c_learns_key():
    ic.EPOCHS, ic.PATIENCE = 60, 10
    D = synth(N=900, mode='key', din=6)
    q = np.arange(9 * HS_, 900)
    qtr, qva = q[:500], q[500:]
    ds = list(range(ic.H, 8 * ic.H + 1, ic.P))
    model, sel, _ = ic.train_c(D, qtr, D, qva, ds, SD, 'own', 'eps', 0)
    ss, o = ic.sse_c(model, D, qva, ds, SD, 'own', 'eps', torch.Generator().manual_seed(0), 2)
    fit = ic.mean_stats(D, qtr, ds, SD, 'own')
    mean = ic.mean_red(fit, ic.mean_stats(D, qva, ds, SD, 'own'))
    real, sh = ic.red(ss, o[0]), np.mean([ic.red(ss, x) for x in o[1:]])
    assert real > mean + 0.3 and real > sh + 0.3, (real, mean, sh)


if __name__ == '__main__':
    test_demo_idx_no_leak(); test_part_a_bias(); test_part_b_ridge_learns_slope(); test_part_c_learns_key(); print('ok')
