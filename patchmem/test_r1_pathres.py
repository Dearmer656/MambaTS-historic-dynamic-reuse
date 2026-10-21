"""PAT-307 R1PathRes checks: (1) at init R1PathRes == R1 exactly; (2) the recurrence equals the closed form
c = sum_j (prod_{i>j} lam_i H_i) u_j with explicit r x r matrices; noH / 'uniform' audit use H = I; (3) the forecast loss reaches
W_o, W_e, W_k, W_b, W_l; (4) zero audit == R1; the correction depends on the lookback errors."""
import sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args, ETTH2
from patchmem.r1_model import R1
from patchmem.r1_pathres import R1PathRes

torch.manual_seed(0)
a = official_args(ETTH2 + ['--des', 'test'])
dev = 'cuda'
r1 = R1(a).to(dev).eval(); pr = R1PathRes(a).to(dev).eval()
pr.load_state_dict(r1.state_dict(), strict=False)
x = torch.randn(4, 720, 7, device=dev)
with torch.no_grad():
    p1, p2 = r1(x, 4)['pred'], pr(x, 4)['pred']
print('init equal', (p1 - p2).abs().max().item()); assert torch.equal(p1, p2)
net = pr.obs_net
with torch.no_grad():                                                                  # closed form vs recurrence
    M, r = 9, net.r
    h = torch.randn(3, M, pr.D, pr.N, device=dev); u = torch.randn(3, M, r, device=dev)
    for noH in (False, True):
        net.noH = noH
        K, B, Lm = net.coef(h)
        c = torch.zeros(3, r, device=dev)
        for t in range(M):
            c = net.step(c, K[:, t], B[:, t], Lm[:, t]) + u[:, t]
        I = torch.eye(r, device=dev)
        Tm = [Lm[:, t, :, None] * (I if noH else I - B[:, t, :, None] * K[:, t, :, None] * K[:, t, None, :]) for t in range(M)]
        ref = torch.zeros(3, r, device=dev)
        for j in range(M):
            Pm = I.expand(3, r, r)
            for i in range(j + 1, M):
                Pm = Tm[i] @ Pm
            ref = ref + (Pm @ u[:, j, :, None]).squeeze(-1)
        print('noH', noH, 'recurrence == closed form', (c - ref).abs().max().item()); assert (c - ref).abs().max() < 1e-4
    net.noH = False
    net.audit = 'uniform'; cu = torch.zeros(3, r, device=dev)
    for t in range(M):
        cu = net.step(cu, K[:, t], B[:, t], Lm[:, t]) + u[:, t]
    net.audit = None
    print('uniform audit == noH', (cu - c).abs().max().item()); assert (cu - c).abs().max() < 1e-5
    net.Wo.weight.normal_(0, 0.1)
    pc = pr(x, 4)['pred']; print('correction changes forecast', (pc - p1).abs().max().item()); assert (pc - p1).abs().max() > 1e-4
    pr.set_audit('zero'); pz = pr(x, 4)['pred']; pr.set_audit(None)
    print('zero audit == R1', (pz - p1).abs().max().item()); assert (pz - p1).abs().max() < 1e-5
    pr.set_audit('uniform'); pu = pr(x, 4)['pred']; pr.set_audit(None)
    print('uniform (H=I) audit changes forecast', (pu - pc).abs().max().item()); assert (pu - pc).abs().max() > 1e-6
    x2 = x.clone(); x2[:, 48:96] += 1.0
    print('lookback dependence', (pr(x2, 4)['pred'] - pc).abs().max().item())
pr.train()
o = pr(x, 3); y = torch.randn_like(o['pred'])
(o['pred'] - y).square().mean().backward()
for n in ('Wo.weight', 'We.weight', 'Wk.weight', 'Wb.weight', 'Wl.weight'):
    g = dict(net.named_parameters())[n].grad
    print('grad', n, None if g is None else g.norm().item()); assert g is not None and g.norm() > 0
print('params', sum(p.numel() for p in net.parameters()), 'stats', pr.stats())
print('OK')
