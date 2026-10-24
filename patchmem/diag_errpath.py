"""PAT-307 ErrPath diagnostic: how much does the value path change the retrieved errors? Per seed/data: mean beta, per-call
relative change ||H v - v|| / ||v||, cumulative change of the k=0 values ||V_path - V|| / ||V|| by slot age, alignment of w with the
errors, the correction change vs the noH audit, and test MSE normal vs noH. usage: python patchmem/diag_errpath.py E|W SEED L H"""
import sys, torch, numpy as np
sys.path.insert(0, '.')
np.Inf = np.inf
from patchmem.mech_eval import DATA
from patchmem.naive_fb_eval import load
D, s, L, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
base, ck, _, _ = DATA[D]
des = ('R1WObK0.5Q8PVh0s{}' if D == 'W' else 'R1ObK0.5Q8PVh0s{}').format(s)
e = load(base, ck, des, s, {'R1_OBS': '1', 'R1_OBS_VPATH': '1', 'R1_MEM_QK_DIM': '8', 'R1_MEM_SLOT_KEEP': '0.5'})
m, dev, P = e.model, e.device, e.model.P
net = m.obs_net
rec = dict(beta=[], rel=[], cos=[])
f0 = net.transport_v
def tv(h, V):
    out = f0(h, V)
    if net.audit != 'noH':
        z = h.detach().flatten(1)
        rec['beta'].append((2 * torch.sigmoid(net.Wbv(z))).mean().item())
        w = torch.nn.functional.normalize(net.Wwv(z), dim=-1)
        Vf = V.flatten(2)
        rec['rel'].append(((out - V).flatten(2).norm(dim=-1) / Vf.norm(dim=-1).clamp_min(1e-8)).mean().item())
        rec['cos'].append((torch.einsum('bmd,bd->bm', Vf, w).abs() / Vf.norm(dim=-1).clamp_min(1e-8)).mean().item())
    return out
net.transport_v = tv
cap = {}
c0 = m._correct
def cor(h, K, V, mask):
    cap.setdefault(net.audit, []).append(V)
    d = c0(h, K, V, mask)
    cap.setdefault(('d', net.audit), []).append(d)
    return d
m._correct = cor
ds = e._ds('test', L, H)
ids = list(range(len(ds)))
se = {None: 0.0, 'noH': 0.0}; nel = 0
cum, dd = [], []
with torch.no_grad():
    for s0 in range(0, len(ids), 64):
        bx, by = e._batch(ds, ids[s0:s0 + 64]); bx, y = bx.to(dev), by[:, -H:].to(dev)
        perm = m.scan_order(bx.shape[0], dev)
        for a in (None, 'noH'):
            cap.clear(); net.audit = a
            se[a] += (m(bx, H // P, perm=perm)['pred'] - y).square().sum().item()
            if a is None:
                Vp, dp = cap[None][0], cap[('d', None)]
            else:
                V0, d0 = cap['noH'][0], cap[('d', 'noH')]
        cum.append(((Vp - V0).flatten(2).norm(dim=-1) / V0.flatten(2).norm(dim=-1).clamp_min(1e-8)).mean(0).cpu())
        dd.append(torch.stack([(a - b).norm() / a.norm().clamp_min(1e-8) for a, b in zip(dp, d0)]).cpu())
        nel += y.numel()
net.audit = None
cum = torch.stack(cum).mean(0).numpy(); dd = torch.stack(dd).mean(0).numpy()
print(f'[errpath {D} s{s} {L}/{H}] dim C*P={m.c * P} | beta mean {np.mean(rec["beta"]):.3f} | per-call ||Hv-v||/||v|| {np.mean(rec["rel"]):.4f} | |cos(v,w)| {np.mean(rec["cos"]):.4f} (random ~{1/np.sqrt(m.c*P):.4f})')
print(f'  k=0 cumulative ||V_path-V||/||V|| by slot (oldest..newest): ' + ' '.join(f'{x:.3f}' for x in cum))
print(f'  ||delta - delta_noH|| / ||delta|| per rollout patch: ' + ' '.join(f'{x:.3f}' for x in dd))
print(f'  test mse normal {se[None]/nel:.4f} | noH {se["noH"]/nel:.4f}')
