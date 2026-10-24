"""PAT-307 ErrPath-e (vpath, vsrc='e') checks: (1) at init R1Obs == R1 exactly; (2) the values read at every rollout patch equal the
explicit products H(e_{M-1}) ... H(e_{j+1}) e_j with H(e) = I - beta(e) w(e) w(e)^T in the C*P space, the SAME at every rollout patch
(H = I in the rollout), keys = O's; (3) a slot masked out (slot keep / lookback) does not transport: its H = I; (4) audit 'noH' == O
with the same weights, bit for bit; (5) the forecast loss reaches W_wv, W_bv, W_q, W_k, W, W_g; W_wv is C*P x C*P (error input)."""
import sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args, ETTH2
from patchmem.r1_model import R1
from patchmem.r1_obsmem import R1Obs

torch.manual_seed(0)
a = official_args(ETTH2 + ['--des', 'test'])
dev = 'cuda'
r1 = R1(a).to(dev).eval(); ov = R1Obs(a, qk_dim=8, slot_keep=0.5, vpath=True, vsrc='e').to(dev).eval()
ov.load_state_dict(r1.state_dict(), strict=False)
b = 3; x = torch.randn(b, 720, 7, device=dev)
perm = ov.scan_order(b, dev)
with torch.no_grad():
    p1, p2 = r1(x, 4, perm=perm)['pred'], ov(x, 4, perm=perm)['pred']
print('1 init equal', (p1 - p2).abs().max().item()); assert torch.equal(p1, p2)
net = ov.obs_net
with torch.no_grad():
    net.W.normal_(0, 0.1); net.Wg.weight.normal_(0, 0.01); net.Wbv.bias.fill_(0.5)
seen = []
f0 = net.forward
net.forward = lambda h, K, V, mask=None: (seen.append((h.clone(), K.clone(), V.clone())), f0(h, K, V, mask))[1]
with torch.no_grad():
    tc = {}; ov(x, 3, perm=perm, trace=tc)
net.forward = f0
T = 720 // 48; M = T - 1
def Hmat(e):
    z = e.flatten(1)
    w = torch.nn.functional.normalize(net.Wwv(z), dim=-1)
    beta = 2 * torch.sigmoid(net.Wbv(z))
    return torch.eye(z.shape[1], device=dev) - beta[..., None] * w[:, :, None] * w[:, None, :]
with torch.no_grad():
    bst = tc['bstates']
    yh = ov.to_orig(ov.read(bst), perm)
    xn = ov.revin(x)[2]
    xo = xn.permute(0, 2, 1).reshape(b, 7, T, 48).permute(0, 2, 1, 3)
    E = xo[:, 1:] - yh[:, :-1]                                                           # (b, M, C, P)
    Ef = E.flatten(2)
    K0 = net.key(bst[:, 1:].flatten(0, 1)).unflatten(0, (b, M))
    Hs = [Hmat(E[:, i]) for i in range(M)]
    ref = torch.empty_like(Ef)
    for j in range(M):
        v = Ef[:, j]
        for s in range(j + 1, M):                                                       # H(e_{j+1}) first, H(e_{M-1}) last
            v = torch.einsum('bde,be->bd', Hs[s], v)
        ref[:, j] = v
    for k in range(3):
        err = (seen[k][2].flatten(2) - ref).abs().max().item(); kerr = (seen[k][1] - K0).abs().max().item()
        print(f'2 rollout patch {k}: values == explicit H(e) products {err:.2e} | keys == O keys {kerr:.2e}'); assert err < 1e-5 and kerr == 0
    assert (ref - Ef).abs().max() > 1e-3                                                 # the path moved the values
    keep = torch.tensor([True, False, True], device=dev)
    Vr = torch.randn(b, 4, 7, 48, device=dev)
    out = net.transport_e(E[:, 5], Vr, keep)
    print('3 masked slot: H = I for the dropped sample', (out[1] - Vr[1]).abs().max().item(), '| kept moved', (out[0] - Vr[0]).abs().max().item())
    assert torch.equal(out[1], Vr[1]) and (out[0] - Vr[0]).abs().max() > 1e-4
    lens = torch.tensor([720, 480, 288], device=dev)                                    # end to end: lookback mask through forward
    seen = []
    net.forward = lambda h, K, V, mask=None: (seen.append(V.clone()), f0(h, K, V, mask))[1]
    tc = {}; ov(x, 2, perm=perm, trace=tc, lens=lens)
    net.forward = f0
    bst = tc['bstates']; yh = ov.to_orig(ov.read(bst), perm)
    xn, t0 = ov.revin(x, lens)[2], ov.revin(x, lens)[3]
    xo = xn.permute(0, 2, 1).reshape(b, 7, T, 48).permute(0, 2, 1, 3)
    E2 = xo[:, 1:] - yh[:, :-1]; Ef2 = E2.flatten(2)
    ok = torch.arange(1, T, device=dev)[None, :] >= (t0 + 1)[:, None]                  # slot i (index) valid
    eye = torch.eye(7 * 48, device=dev)
    Hs2 = [torch.where(ok[:, i, None, None], Hmat(E2[:, i]), eye) for i in range(M)]
    ref2 = torch.empty_like(Ef2)
    for j in range(M):
        v = Ef2[:, j]
        for s_ in range(j + 1, M):
            v = torch.einsum('bde,be->bd', Hs2[s_], v)
        ref2[:, j] = v
    err2 = max((sv.flatten(2) - ref2).abs().max().item() for sv in seen)
    print(f'3b lens mask {lens.tolist()} t0 {t0.tolist()}: values == products with H = I on masked slots {err2:.2e}'); assert err2 < 1e-5
    o = R1Obs(a, qk_dim=8, slot_keep=0.5).to(dev).eval()
    o.load_state_dict({k: v for k, v in ov.state_dict().items() if not k.startswith(('obs_net.Wwv', 'obs_net.Wbv'))})
    ov.set_audit('noH'); pn = ov(x, 4, perm=perm)['pred']; ov.set_audit(None)
    po = o(x, 4, perm=perm)['pred']; pp = ov(x, 4, perm=perm)['pred']
    print('4 noH audit == O', (pn - po).abs().max().item(), '| path changes forecast', (pp - po).abs().max().item())
    assert torch.equal(pn, po) and (pp - po).abs().max() > 1e-6
ov.train()
out = ov(x, 3); y = torch.randn_like(out['pred'])
(out['pred'] - y).square().mean().backward()
for n in ('W', 'Wq.weight', 'Wk.weight', 'Wg.weight', 'Wwv.weight', 'Wbv.weight'):
    g = dict(net.named_parameters())[n].grad
    print('5 grad', n, None if g is None else g.norm().item()); assert g is not None and g.norm() > 0
assert net.Wwv.weight.shape == (7 * 48, 7 * 48), net.Wwv.weight.shape
print('params', sum(p.numel() for p in net.parameters()))
print('OK')
