"""PAT-304 F3 (A0Corr mode='cross') tests (GPU). usage: python patchmem/test_a0corr_cross.py <A0 checkpoint.pth>"""
import sys
from argparse import Namespace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.MambaTS import Model
from patchmem.a0corr import A0Corr

dev = 'cuda'
C, P, NP, H = 7, 48, 15, 96
cfg = Namespace(task_name='long_term_forecast', seq_len=P * NP, pred_len=H, patch_len=P, stride=P, d_model=16, e_layers=5, enc_in=C, dropout=0.3,
                VPT_mode=1, ATSP_solver='SA', use_casual_conv=False)
order = [6, 2, 1, 5, 0, 4, 3]


def ok(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg); assert cond, msg


def build(shuffle=False):
    base = Model(cfg).float(); base.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
    return A0Corr(base, order, corr='shared', shuffle=shuffle, mode='cross').to(dev)


torch.manual_seed(0)
x = torch.randn(4, P * NP, C, device=dev).cumsum(1) * 0.1
m = build(); m.train()
with torch.no_grad():
    m.ro.weight.normal_(0, 0.02); m.ro.bias.normal_(0, 0.1)
ref = m.base; ref.encoder.ids_shuffle = torch.tensor(order, device=dev)[None].repeat(x.shape[0], 1)

print('T1 zero correction == official A0; per-step loss == uncorrected readout MSE')
with torch.no_grad():
    y0 = ref(x.clone(), None, None, None)
    rec = []; o = m.run_cross(x, record=rec)
    h, _, xp = m.scan_states(x)
    s0 = (xp[:, :, 1:] - m.ro(h[:, :, :-1])).square().mean().item()
ok((o['pred'] - y0).abs().max().item() < 1e-4, f'pred max diff {(o["pred"] - y0).abs().max().item():.1e}')
ok(abs(o['step'].item() - o['step0'].item()) < 1e-5 and abs(o['step0'].item() - s0) < 1e-5, f"step {o['step'].item():.5f} step0 {o['step0'].item():.5f} ref {s0:.5f}")
ok(len(rec) == C * (NP - 1), f'{len(rec)} correction sites')

print('T2 input u at (k, t) = [e*_{k,t}, e*_{j,t+1} for j < k, zeros]')
with torch.no_grad():
    es = torch.cat([torch.zeros_like(xp[:, :, :1]), xp[:, :, 1:] - m.ro(h[:, :, :-1])], 2)
    err = 0.0
    for s, u in rec:
        k, t = divmod(s, NP)
        u = u.reshape(x.shape[0], C, P)
        err = max(err, (u[:, 0] - es[:, k, t]).abs().max().item(), (u[:, 1:k + 1] - es[:, :k, t + 1]).abs().max().item() if k else 0.0,
                  u[:, k + 1:].abs().max().item() if k < C - 1 else 0.0)
ok(err < 1e-5, f'max diff {err:.1e}')

print('T3 scan causality: perturbing the variable at scan position 3 leaves the inputs of sites k < 3 unchanged')
x2 = x.clone(); x2[:, :, order[3]] += torch.randn_like(x2[:, :, order[3]])
with torch.no_grad():
    rec2 = []; m.run_cross(x2, record=rec2)
d = [(u - u2).abs().max().item() for (s, u), (_, u2) in zip(rec, rec2)]
ok(max(d[:3 * (NP - 1)]) < 1e-6 and max(d[3 * (NP - 1):]) > 1e-4, f'max diff k<3 {max(d[:3 * (NP - 1)]):.1e}, k>=3 {max(d[3 * (NP - 1):]):.2f}')

print('T4 nonzero correction changes the forecast; shuffle differs; gradients only on CorrNet')
with torch.no_grad():
    m.corr_net.W2.normal_(0, 0.05); y1 = m(x)
ok((y1 - y0).abs().max().item() > 1e-4, f'max change {(y1 - y0).abs().max().item():.3f}')
ms = build(shuffle=True); ms.load_state_dict(m.state_dict())
with torch.no_grad():
    ok((ms(x) - y1).abs().max().item() > 1e-5, 'shuffled input gives a different forecast')
m.zero_grad(); out = m.run_cross(x); (out['pred'].square().mean() + out['step'] + 1e-4 * out['reg']).backward()
g = sum(0.0 if p.grad is None else p.grad.abs().sum().item() for p in m.corr_net.parameters())
ok(g > 0 and all(p.grad is None for p in m.base.parameters()) and m.ro.weight.grad is None, f'|grad CorrNet| {g:.2e}, frozen A0/ro untouched')
print(f'   CorrNet params {sum(p.numel() for p in m.corr_net.parameters())}')
print('ALL PASS')
