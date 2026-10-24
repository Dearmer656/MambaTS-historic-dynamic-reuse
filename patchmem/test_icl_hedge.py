"""Unit tests for patchmem/icl_hedge.py: vectorised hedge_all == a naive per-config loop with explicit reveal sets; no lookahead;
carried memory; experts end to end."""
import math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import patchmem.icl_lag as IL
IL.dev = 'cpu'
import patchmem.icl_hedge as M
M.dev = 'cpu'
torch.manual_seed(0); dt = torch.float64
N, K, C, P, D = 40, 5, 3, 4, 3
pr = torch.randn(N, K, C, P); pr[:, 0] = 0
r = torch.randn(N, C, P)
loss = ((r[:, None] - pr) ** 2).mean(-1).double()


def naive(loss, pr, r, delay, half, eta, share, L0=None):
    g = 1.0 if math.isinf(half) else 0.5 ** (1 / half)
    sse = 0.0
    for i in range(N):
        L = torch.zeros(K, C, dtype=dt) if L0 is None else L0.clone() * g ** (max(i - delay + 1, 0))
        for j in range(0, i - delay + 1):
            L += g ** (i - delay - j) * loss[j]
        if not bool((L[0] > 0).all()):
            W = torch.full((K, C), 1 / K, dtype=dt)
        else:
            Ls = L.sum(-1, keepdim=True).expand(K, C) if share else L
            rel = Ls / Ls[0:1]; rel = rel - rel.min(0).values
            W = torch.nn.functional.one_hot(rel.argmin(0), K).T.to(dt) if math.isinf(eta) else torch.softmax(-eta * rel, 0)
        d = torch.einsum('kc,kcp->cp', W, pr[i].double())
        sse += ((r[i].double() - d) ** 2).sum().item()
    return sse


sse, Lend = M.hedge_all(loss, pr, r, D)
for c, (h, e, s) in enumerate(M.CFGS):
    ref = naive(loss, pr, r, D, h, e, s)
    assert abs(sse[c].item() - ref) < 1e-8 * max(1, ref), (h, e, s, sse[c].item(), ref)
# carried memory (test continues from val): naive with L0 decayed per reveal step
L0 = Lend
sse2, _ = M.hedge_all(loss, pr, r, D, L0)
for c, (h, e, s) in enumerate(M.CFGS):
    ref = naive(loss, pr, r, D, h, e, s, L0[M.HALF.index(h)])
    assert abs(sse2[c].item() - ref) < 1e-8 * max(1, ref), ('carry', h, e, s, sse2[c].item(), ref)
# no lookahead: changing the loss of window j must not change any prediction at windows i < j + D
for j in (5, 20):
    l2 = loss.clone(); l2[j] += 10 * torch.rand(K, C, dtype=dt)
    for upto in (j + D,):
        a, _ = M.hedge_all(loss[:upto], pr[:upto], r[:upto], D); b, _ = M.hedge_all(l2[:upto], pr[:upto], r[:upto], D)
        assert torch.allclose(a, b), 'lookahead'
    a, _ = M.hedge_all(loss[:j + D + 1], pr[:j + D + 1], r[:j + D + 1], D); b, _ = M.hedge_all(l2[:j + D + 1], pr[:j + D + 1], r[:j + D + 1], D)
    assert not torch.allclose(a, b), 'outcome never used'
# experts end to end on a periodic error sequence: single lag g0 expert ~ perfect
J, g0 = 29, 5
cs = torch.randn(g0, 2, 2, dtype=dt); tt = torch.arange(IL.P, dtype=dt) - (IL.P - 1) / 2
base = cs[:, 0, None] + cs[:, 1, None] * tt[None, :, None] * 0.05
E = torch.stack([base[t % g0] for t in range(J + IL.NP)], 0)[None].repeat(3, 1, 1, 1)
Fut = E[:, J:].reshape(3, -1, 2).numpy(); Hs = E[:, :J].flip(1).numpy()
gen = torch.Generator().manual_seed(0)
names, rr, preds = M.expert_preds(Fut, Hs, 2, gen, 2, 1)
i = names.index('single5')
rel = (((preds[0][:, i] - rr) ** 2).mean() / (rr ** 2).mean()).item(); print('single5 rel mse', rel, 'single6', (((preds[0][:, names.index('single6')] - rr) ** 2).mean() / (rr ** 2).mean()).item())
assert rel < 0.2 and names[:5] == ['zero', 'mean', 'recent4', 'recent8', 'recent12']
assert torch.equal(preds[0][:, :5], preds[1][:, :5]), 'null leaves input-free experts unchanged'
print('ALL PASS', len(names), 'experts')
