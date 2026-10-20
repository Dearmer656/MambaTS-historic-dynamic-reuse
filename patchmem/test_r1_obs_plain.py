"""PAT-307 O-plain checks: (1) at init R1Obs(plain) == R1 exactly; (2) one head, the correction equals gamma * sum_j a_j e_j recomputed
by hand from the returned attention and the lookback errors; (3) the forecast loss reaches gamma, W_Q, W_K, tau."""
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
r1 = R1(a).to(dev).eval(); ob = R1Obs(a, slot_keep=0.5, plain=True).to(dev).eval()
ob.load_state_dict(r1.state_dict(), strict=False)
assert ob.obs_net.nh == 1 and ob.obs_net.W.numel() == 1 and not hasattr(ob.obs_net, 'Wg')
x = torch.randn(4, 720, 7, device=dev)
with torch.no_grad():
    p1, p2 = r1(x, 4)['pred'], ob(x, 4)['pred']
print('init equal', (p1 - p2).abs().max().item()); assert torch.equal(p1, p2)
with torch.no_grad():
    ob.obs_net.W.fill_(0.7)
    b, M, C, P = 4, 14, 7, 48
    h = torch.randn(b, ob.D, ob.N, device=dev)
    keys = torch.randn(b, M, 1, 8, device=dev); vals = torch.randn(b, M, C, P, device=dev)
    d, att, g = ob.obs_net(h, keys, vals)
    ref = 0.7 * torch.einsum('bm,bmcp->bcp', att[:, 0], vals)
    print('delta == gamma * sum a e', (d - ref).abs().max().item()); assert (d - ref).abs().max() < 1e-5
    pc = ob(x, 4)['pred']; print('correction changes forecast', (pc - p1).abs().max().item()); assert (pc - p1).abs().max() > 1e-4
ob.train()
o = ob(x, 3); y = torch.randn_like(o['pred'])
(o['pred'] - y).square().mean().backward()
for n in ('W', 'Wq.weight', 'Wk.weight', 'log_tau'):
    gr = dict(ob.obs_net.named_parameters())[n].grad
    print('grad', n, None if gr is None else gr.norm().item()); assert gr is not None and gr.norm() > 0
print('params', sum(p.numel() for p in ob.obs_net.parameters()))
print('OK')
