"""PAT-307 O-PathV-shared (vshare) checks: (1) at init R1Obs(vpath, vshare) == R1 exactly; (2) the values used at rollout patch k equal
the explicit products H_{s_k} ... H_{j+1} e_j with H = I_C (x) H_P (the same P x P Householder for every variable), captured inside forward;
the keys are O's; (3) no mixing across variables: transport_v of variable c ignores the other variables' errors; (4) audit 'noH' == an O
model with the same weights, bit for bit; (5) the forecast loss reaches W_wv, W_bv (and W_q, W_k, W, W_g); W_wv is d_inner*d_state x P."""
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
r1 = R1(a).to(dev).eval(); ov = R1Obs(a, qk_dim=8, slot_keep=0.5, vpath=True, vshare=True).to(dev).eval()
ov.load_state_dict(r1.state_dict(), strict=False)
b = 3; x = torch.randn(b, 720, 7, device=dev)
perm = ov.scan_order(b, dev)
with torch.no_grad():
    p1, p2 = r1(x, 4, perm=perm)['pred'], ov(x, 4, perm=perm)['pred']
print('init equal', (p1 - p2).abs().max().item()); assert torch.equal(p1, p2)
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
with torch.no_grad():
    bst = tc['bstates']
    yh = ov.to_orig(ov.read(bst), perm)
    xn = ov.revin(x)[2]
    xo = xn.permute(0, 2, 1).reshape(b, 7, T, 48).permute(0, 2, 1, 3)
    E = (xo[:, 1:] - yh[:, :-1]).flatten(2)                                              # (b, M, C*P) raw errors
    K0 = net.key(bst[:, 1:].flatten(0, 1)).unflatten(0, (b, M))
    def Hmat(h):
        z = h.flatten(1)
        w = torch.nn.functional.normalize(net.Wwv(z), dim=-1)
        beta = 2 * torch.sigmoid(net.Wbv(z))
        HP = torch.eye(w.shape[1], device=dev) - beta[..., None] * w[:, :, None] * w[:, None, :]       # (b, P, P)
        return torch.stack([torch.kron(torch.eye(7, device=dev), HP[i]) for i in range(HP.shape[0])])   # (b, C*P, C*P), C-major
    Hs = [Hmat(bst[:, i + 1]) for i in range(M)] + [Hmat(seen[k][0]) for k in (1, 2)]
    for k in range(3):
        last = M - 1 + k
        ref = torch.empty_like(E)
        for j in range(M):
            v = E[:, j]
            for s in range(j + 1, last + 1):
                v = torch.einsum('bde,be->bd', Hs[s], v)
            ref[:, j] = v
        err = (seen[k][2].flatten(2) - ref).abs().max().item()
        kerr = (seen[k][1] - K0).abs().max().item()
        print(f'rollout patch {k}: values == explicit H products {err:.2e} | keys == O keys {kerr:.2e}'); assert err < 1e-5 and kerr == 0
    assert (seen[2][2] - seen[0][2]).abs().max() > 1e-4                                  # the path moved the values
    V = torch.randn(b, M, 7, 48, device=dev); V2 = V.clone(); V2[:, :, 1:] = torch.randn_like(V2[:, :, 1:])
    d0 = (net.transport_v(bst[:, 2], V)[:, :, 0] - net.transport_v(bst[:, 2], V2)[:, :, 0]).abs().max().item()
    print('variable 0 ignores the other variables', d0); assert d0 == 0
    assert net.Wwv.weight.shape == (48, bst[:, 0].numel() // b), net.Wwv.weight.shape
    o = R1Obs(a, qk_dim=8, slot_keep=0.5).to(dev).eval()
    o.load_state_dict({k: v for k, v in ov.state_dict().items() if not k.startswith(('obs_net.Wwv', 'obs_net.Wbv'))})
    ov.set_audit('noH'); pn = ov(x, 4, perm=perm)['pred']; ov.set_audit(None)
    po = o(x, 4, perm=perm)['pred']; pp = ov(x, 4, perm=perm)['pred']
    print('noH audit == O', (pn - po).abs().max().item()); assert torch.equal(pn, po)
    print('path changes forecast', (pp - po).abs().max().item()); assert (pp - po).abs().max() > 1e-6
ov.train()
out = ov(x, 3); y = torch.randn_like(out['pred'])
(out['pred'] - y).square().mean().backward()
for n in ('W', 'Wq.weight', 'Wk.weight', 'Wg.weight', 'Wwv.weight', 'Wbv.weight'):
    g = dict(net.named_parameters())[n].grad
    print('grad', n, None if g is None else g.norm().item()); assert g is not None and g.norm() > 0
print('params', sum(p.numel() for p in net.parameters()))
print('OK')
