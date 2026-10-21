"""PAT-307 O-PaTH checks: (1) at init R1Obs(path) == R1 exactly; (2) the stored keys used at rollout patch k equal the explicit
products H_{s_k} ... H_{j+1} k_j (per head, s = slot states then rollout states), captured from inside forward; (3) audit 'noH' ==
an O model (path=False) with the same weights, bit for bit; (4) the forecast loss reaches W_w and W_b (and W_q, W_k, W, W_g)."""
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
r1 = R1(a).to(dev).eval(); op = R1Obs(a, qk_dim=8, slot_keep=0.5, path=True).to(dev).eval()
op.load_state_dict(r1.state_dict(), strict=False)
x = torch.randn(3, 720, 7, device=dev)
perm = op.scan_order(3, dev)
with torch.no_grad():
    p1, p2 = r1(x, 4, perm=perm)['pred'], op(x, 4, perm=perm)['pred']
print('init equal', (p1 - p2).abs().max().item()); assert torch.equal(p1, p2)
net = op.obs_net
with torch.no_grad():
    net.W.normal_(0, 0.1); net.Wg.weight.normal_(0, 0.01); net.Wb.bias.fill_(0.5)
# capture keys / query states seen by the memory at each rollout patch, and every transport call
seen, hs = [], []
f0, t0 = net.forward, net.transport
net.forward = lambda h, K, V, mask=None: (seen.append((h.clone(), K.clone())), f0(h, K, V, mask))[1]
net.transport = lambda h, K: (hs.append(h.clone()), t0(h, K))[1]
with torch.no_grad():
    op(x, 3, perm=perm)
net.forward, net.transport = f0, t0
T = 720 // 48; M = T - 1
assert len(seen) == 3 and len(hs) == (M - 1) + 2, (len(seen), len(hs))
with torch.no_grad():
    tr = op.trace if hasattr(op, 'trace') else None
    tc = {}; op(x, 3, perm=perm, trace=tc)
    xn = op.revin(x)[2]
    K0 = None
    # raw keys k_j from the boundary states (slot j uses bstates[:, j+1])
    bst = tc['bstates']
    K0 = net.key(bst[:, 1:].flatten(0, 1)).unflatten(0, (3, M))                       # (b, M, nh, dh)
    def Hmat(h):
        z = h.flatten(1)
        w = torch.nn.functional.normalize(net.Ww(z).reshape(-1, net.nh, net.dh), dim=-1)
        beta = 2 * torch.sigmoid(net.Wb(z))
        return torch.eye(net.dh, device=dev) - beta[..., None, None] * w[..., :, None] * w[..., None, :]   # (b, nh, dh, dh)
    Hs = [Hmat(bst[:, i + 1]) for i in range(M)] + [Hmat(seen[k][0]) for k in (1, 2)]   # slot states, then rollout states
    for k in range(3):
        last = M - 1 + k                                                                 # index of the newest transition applied
        ref = torch.empty_like(K0)
        for j in range(M):
            v = K0[:, j]                                                                 # (b, nh, dh)
            for s in range(j + 1, last + 1):
                v = torch.einsum('bhde,bhe->bhd', Hs[s], v)
            ref[:, j] = v
        err = (seen[k][1] - ref).abs().max().item()
        print(f'rollout patch {k}: keys == explicit H products', err); assert err < 1e-5
        assert (seen[k][0] - (bst[:, -1] if k == 0 else seen[k][0])).abs().max() == 0
    # (3) noH audit == O with the same weights
    o = R1Obs(a, qk_dim=8, slot_keep=0.5, path=False).to(dev).eval()
    o.load_state_dict({k: v for k, v in op.state_dict().items() if not k.startswith(('obs_net.Ww', 'obs_net.Wb'))})
    op.set_audit('noH'); pn = op(x, 4, perm=perm)['pred']; op.set_audit(None)
    po = o(x, 4, perm=perm)['pred']; pp = op(x, 4, perm=perm)['pred']
    print('noH audit == O', (pn - po).abs().max().item()); assert torch.equal(pn, po)
    print('path changes forecast', (pp - po).abs().max().item()); assert (pp - po).abs().max() > 1e-6
op.train()
out = op(x, 3); y = torch.randn_like(out['pred'])
(out['pred'] - y).square().mean().backward()
for n in ('W', 'Wq.weight', 'Wk.weight', 'Wg.weight', 'Ww.weight', 'Wb.weight'):
    g = dict(net.named_parameters())[n].grad
    print('grad', n, None if g is None else g.norm().item()); assert g is not None and g.norm() > 0
print('params', sum(p.numel() for p in net.parameters()))
print('OK')
