"""PAT-304 C-series unit tests: residual-driven state correction on the readout layer (causal head). usage: python patchmem/test_corr.py [full|next]"""
import sys
from argparse import Namespace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.MambaTS import Model
from patchmem.tm_model import TMMambaTS

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
C, P, NP, H = 7, 48, 6, 96
RES = sys.argv[1] if len(sys.argv) > 1 else 'full'


def build(corr=None, shuffle=False, seed=0):
    torch.manual_seed(seed)
    cfg = Namespace(task_name='long_term_forecast', seq_len=P * NP, pred_len=H, patch_len=P, stride=P, d_model=16, e_layers=2,
                    enc_in=C, dropout=0.0, VPT_mode=1, ATSP_solver='SA', use_casual_conv=False)
    return TMMambaTS(Model(cfg), head='causal', detach_trunk=True, corr=corr, corr_shuffle=shuffle, corr_res=RES).to(dev).eval()


def ok(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg); assert cond, msg


torch.manual_seed(1)
b = 4
x = torch.randn(b, P * NP, C, device=dev)
perm = torch.stack([torch.randperm(C) for _ in range(b)]).to(dev)
base = build()
m = build('shared'); m.load_state_dict(base.state_dict(), strict=False)
d, jmax, L = m.corr_delay, m.corr_jmax, m.corr_res_len
print(f'res {RES}: delay {d}, residual length {L}, corrections at boundaries {d}..{jmax + d}')

print('T1 zero-initialised correction == no correction (E6)')
with torch.no_grad():
    o0, o1 = base(x, perm), m(x, perm)
ok((o0['pred'] - o1['pred']).abs().max().item() < 1e-5, f"pred max diff {(o0['pred'] - o1['pred']).abs().max().item():.2e}")
ok(o1['corr_reg'].item() == 0, 'regulariser is 0 at init')

print('T2 residual fed to CorrNet == Y_j - (normalised reported pred_j)')
rec = []
fwd = m.corr_net.forward
m.corr_net.forward = lambda h, e: (rec.append(e), fwd(h, e))[1]
with torch.no_grad():
    o = m(x, perm)
m.corr_net.forward = fwd
means, stdev = o['stats']
xn = (x - means) / stdev
for j, e in enumerate(rec):
    pn = (o['pred'][:, j] - means) / stdev
    ref = (xn[:, P * (j + 1): P * (j + 1) + L] - pn[:, :L]).reshape(b, -1)
    ok((e - ref).abs().max().item() < 1e-4, f'e_{j} matches (max diff {(e - ref).abs().max().item():.1e})')

print('T3 nonzero correction: origins <= delay unchanged, later origins change')
with torch.no_grad():
    m.corr_net.W2.normal_(0, 0.5); m.corr_net.b2.normal_(0, 0.5)
    o2 = m(x, perm)
diff = (o2['pred'] - o1['pred']).abs().amax((0, 2, 3))
ok(torch.all(diff[:d + 1] < 1e-6).item(), f'pred[:, :{d + 1}] unchanged {diff[:d + 1].tolist()}')
ok(torch.all(diff[d + 1:] > 1e-5).item(), f'pred[:, {d + 1}:] changed {[round(v, 4) for v in diff[d + 1:].tolist()]}')

print('T4 causality (window stats fixed): perturbing x from patch k leaves pred[:, :k] unchanged')
for k in (2, 3, 4):
    with torch.no_grad():
        x2 = x.clone(); x2[:, P * k:] += torch.randn_like(x2[:, P * k:])
        o3 = m(x2, perm, stats=o2['stats'])
    ok((o3['pred'][:, :k] - o2['pred'][:, :k]).abs().max().item() < 1e-5 and (o3['pred'][:, k:] - o2['pred'][:, k:]).abs().max().item() > 1e-4,
       f'k={k}: max diff on pred[:, :{k}] {(o3["pred"][:, :k] - o2["pred"][:, :k]).abs().max().item():.1e}')

print('T5 gradients')
m.zero_grad(); o4 = m(x, perm); (o4['pred'][:, -1].square().mean()).backward()
g = lambda p: 0.0 if p.grad is None else p.grad.abs().sum().item()
ok(g(m.corr_net.W1) > 0 and g(m.corr_net.E[0].weight) > 0, f'last-origin loss reaches CorrNet (|dW1| {g(m.corr_net.W1):.2e}, |dE| {g(m.corr_net.E[0].weight):.2e})')
ok(g(m.base.value_embedding[1].weight) > 0, 'last-origin loss reaches the trunk through the corrected state (not detached)')
m.zero_grad(); o5 = m(x, perm); (o5['pred'][:, 3].square().mean()).backward()
ok(g(m.corr_net.W1) == 0 and g(m.causal_W) == 0 and g(m.base.value_embedding[1].weight) == 0 and g(m.causal_gain) > 0,
   'detach_trunk: an intermediate-origin loss reaches only gain/bias')
m.zero_grad(); o6 = m(x, perm); (o6['pred'][:, -1].square().mean() + o6['corr_reg']).backward()
ok(g(m.corr_net.W2) > 0, 'regulariser is differentiable w.r.t. CorrNet')

print('T6 shuffle control changes the correction; modes and parameter counts')
ms = build('shared', shuffle=True); ms.load_state_dict(m.state_dict())
with torch.no_grad():
    ok((ms(x, perm)['pred'][:, d + 1:] - o2['pred'][:, d + 1:]).abs().max().item() > 1e-5, 'shuffled residual gives a different prediction')
for mode in ('shared', 'shared_id', 'channel'):
    mm = build(mode)
    print(f'   {mode:10s} CorrNet params {sum(p.numel() for p in mm.corr_net.parameters())}')
    with torch.no_grad():
        mm(x, perm)
print('ALL PASS')
