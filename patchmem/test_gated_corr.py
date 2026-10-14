"""PAT-304 GatedCorrNet / channel_decor / A0Corr(corr='gated') tests (GPU). usage: python patchmem/test_gated_corr.py"""
import sys
from argparse import Namespace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.MambaTS import Model
from patchmem.a0corr import A0Corr
from patchmem.tm_model import GatedCorrNet, channel_decor

dev = 'cuda'
C, P, NP, H = 5, 48, 15, 96
cfg = Namespace(task_name='long_term_forecast', seq_len=P * NP, pred_len=H, patch_len=P, stride=P, d_model=16, e_layers=2, enc_in=C, dropout=0.1,
                VPT_mode=1, ATSP_solver='SA', use_casual_conv=False)
order = [3, 0, 4, 1, 2]


def ok(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg); assert cond, msg


torch.manual_seed(0)
d_inner, d_state, res_dim = 32, 16, C * P

print('T1 GatedCorrNet: C == 0 at init; C(h, e=0) == 0 for any h; with e != 0 the gate makes C depend on h')
g = GatedCorrNet(d_inner, d_state, res_dim).to(dev)
h = torch.randn(4, d_inner, d_state, device=dev); e = torch.randn(4, res_dim, device=dev)
ok(g(h, e).abs().max().item() == 0.0, 'C == 0 at init (Wc zero-initialised)')
with torch.no_grad():
    g.Wc.normal_(0, 0.1)
c0 = g(h, torch.zeros_like(e)); c0b = g(torch.randn_like(h) * 5, torch.zeros_like(e))
ok(c0.abs().max().item() == 0.0 and c0b.abs().max().item() == 0.0, 'C(h, e=0) == 0 exactly, for two different h')
d_h = (g(h, e) - g(torch.randn_like(h), e)).abs().max().item()
ok(d_h > 1e-4, f'same e, different h -> different C via the gate (max diff {d_h:.3e})')

print('T2 channel_decor: high when C is a function of h, ~0 when C is independent noise, finite gradient at C == 0')
hN = torch.randn(4000, d_inner, d_state, device=dev)
dec_same = channel_decor(hN, hN.clone()).item()
dec_lin = channel_decor(hN, torch.einsum('ndi,ij->ndj', hN, torch.randn(d_state, d_state, device=dev))).item()
dec_noise = channel_decor(hN, torch.randn_like(hN)).item()
ok(dec_same > 0.05 and dec_lin > 0.02 and dec_noise < 1e-3, f'decor: C=h {dec_same:.4f}, C=hM {dec_lin:.4f}, C=noise {dec_noise:.2e}')
cz = torch.zeros_like(hN, requires_grad=True); channel_decor(hN, cz).backward()
ok(torch.isfinite(cz.grad).all().item(), 'gradient finite at C == 0')

print('T3 A0Corr(corr=gated, cross): zero correction at init == official A0 forward; decor == 0')
base = Model(cfg).float().to(dev)
m = A0Corr(base, order, corr='gated', mode='cross').to(dev); m.train()
with torch.no_grad():
    m.ro.weight.normal_(0, 0.02); m.ro.bias.normal_(0, 0.1)
ref = m.base; ref.encoder.ids_shuffle = torch.tensor(order, device=dev)[None].repeat(4, 1)
x = torch.randn(4, P * NP, C, device=dev).cumsum(1) * 0.1
with torch.no_grad():
    y0 = ref(x.clone(), None, None, None); o = m.run_cross(x)
ok((o['pred'] - y0).abs().max().item() < 1e-4, f'pred max diff {(o["pred"] - y0).abs().max().item():.1e}')
ok(o['decor'].item() == 0.0 and o['reg'].item() == 0.0, f'decor {o["decor"].item():.2e}, reg {o["reg"].item():.2e} at init')

print('T4 nonzero correction: changes the forecast, decor in [0, 1], loss with decor trains only the CorrNet (finite grads)')
with torch.no_grad():
    m.corr_net.Wc.normal_(0, 0.05)
m.zero_grad(); o = m.run_cross(x)
ok((o['pred'] - y0).abs().max().item() > 1e-4, f'forecast changes (max {(o["pred"] - y0).abs().max().item():.3f})')
ok(0.0 <= o['decor'].item() <= 1.0, f'decor {o["decor"].item():.4f}')
(o['pred'].square().mean() + o['step'] + 1e-4 * o['reg'] + 0.1 * o['decor']).backward()
gs = [p.grad for p in m.corr_net.parameters()]
ok(all(gr is not None and torch.isfinite(gr).all() for gr in gs) and sum(gr.abs().sum().item() for gr in gs) > 0, 'finite, non-zero grads on every CorrNet param')
ok(all(p.grad is None for p in m.base.parameters()) and m.ro.weight.grad is None, 'frozen A0 / ro untouched')
print(f'   GatedCorrNet params {sum(p.numel() for p in m.corr_net.parameters())}')
print('ALL PASS')
