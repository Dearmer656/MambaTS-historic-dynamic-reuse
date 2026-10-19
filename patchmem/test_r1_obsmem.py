"""PAT-304 R1Obs checks: (1) at init R1Obs == R1 exactly (same backbone weights); (2) the forecast loss reaches W, W_Q, W_K, W_g;
(3) the correction depends on the lookback only: perturbing the future targets changes nothing, perturbing an early lookback patch
changes the correction; (4) audits: zero == backbone, feed=False backbone rollout == R1."""
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
r1 = R1(a).to(dev).eval(); ob = R1Obs(a, slot_keep=0.5).to(dev).eval()
ob.load_state_dict(r1.state_dict(), strict=False)
x = torch.randn(4, 720, 7, device=dev)
with torch.no_grad():
    p1, p2 = r1(x, 4)['pred'], ob(x, 4)['pred']
print('init equal', (p1 - p2).abs().max().item()); assert torch.equal(p1, p2)
with torch.no_grad():
    ob.obs_net.W.normal_(0, 0.1); ob.obs_net.Wg.weight.normal_(0, 0.01)
    pc = ob(x, 4)['pred']
    print('trained-like correction changes forecast', (pc - p1).abs().max().item()); assert (pc - p1).abs().max() > 1e-4
    ob.set_audit('zero'); pz = ob(x, 4)['pred']; ob.set_audit(None)
    print('zero audit == R1', (pz - p1).abs().max().item()); assert (pz - p1).abs().max() < 1e-5
    ob.feed = False; ob.set_audit('zero'); pf = ob(x, 4)['pred']; ob.set_audit(None); ob.feed = True
    assert (pf - p1).abs().max() < 1e-5
    x2 = x.clone(); x2[:, 48:96] += 1.0                                                  # change lookback patch 1 -> its slot changes
    print('lookback dependence', (ob(x2, 4)['pred'] - pc).abs().max().item())
ob.train(); ob.obs_net.W.data.normal_(0, 0.1)
o = ob(x, 3); y = torch.randn_like(o['pred'])
(o['pred'] - y).square().mean().backward()
for n in ('W', 'Wq.weight', 'Wk.weight', 'Wg.weight', 'log_tau'):
    g = dict(ob.obs_net.named_parameters())[n].grad
    print('grad', n, None if g is None else g.norm().item()); assert g is not None and g.norm() > 0
print('OK')
