"""PAT-304 A0Corr(freeze=False) smoke test: backbone + ro train from scratch alongside GatedCorrNet. usage: python patchmem/test_from_scratch.py"""
import sys
from argparse import Namespace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.MambaTS import Model
from patchmem.a0corr import A0Corr

dev = 'cuda'
C, P, NP, H = 5, 48, 15, 96
cfg = Namespace(task_name='long_term_forecast', seq_len=P * NP, pred_len=H, patch_len=P, stride=P, d_model=16, e_layers=2, enc_in=C, dropout=0.1,
                VPT_mode=1, ATSP_solver='SA', use_casual_conv=False)
def ok(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg); assert cond, msg


torch.manual_seed(0)
base = Model(cfg).float().to(dev)
base.encoder.eval()
with torch.no_grad():
    base.encoder(torch.zeros(1, cfg.enc_in * (cfg.seq_len // cfg.patch_len), cfg.d_model, device=dev))   # fix the ATSP scan order once, shared with A0Corr
order = base.encoder.ids_shuffle[0].tolist()
base.encoder.train()
m = A0Corr(base, order, corr='gated', mode='cross', freeze=False).to(dev)

print('T1 freeze=False: backbone and ro require grad, train() leaves the backbone in train mode')
ok(all(p.requires_grad for p in m.base.parameters()), 'all backbone params require grad')
ok(m.ro.weight.requires_grad and m.ro.bias.requires_grad, 'ro requires grad')
m.train()
ok(m.base.training and m.base.encoder.training, 'backbone left in train() mode (not forced to eval)')

print('T2 forward/backward: gradients reach backbone, ro and corr_net together; corr starts at 0 (Wc zero-init) so pred == official forward at init')
x = torch.randn(3, cfg.seq_len, C, device=dev).cumsum(1) * 0.1
m.eval()                                                                      # disable dropout for the exact-equality check only
with torch.no_grad():
    y_ref = base(x.clone(), None, None, None)
    o = m.run_cross(x)
ok((o['pred'] - y_ref).abs().max().item() < 1e-4, f'pred max diff {(o["pred"] - y_ref).abs().max().item():.1e} (Wc==0 at init)')
m.train()
m.zero_grad()
out = m.run_cross(x)
loss = out['pred'].square().mean() + out['step'] + 1e-4 * out['reg'] + 0.1 * out['decor']
loss.backward()
gb = sum(p.grad.abs().sum().item() for p in m.base.parameters() if p.grad is not None)
gr = sum(p.grad.abs().sum().item() for p in [m.ro.weight, m.ro.bias] if p.grad is not None)
gc = sum(p.grad.abs().sum().item() for p in m.corr_net.parameters() if p.grad is not None)
ok(gb > 0, f'|grad backbone| {gb:.3e}')
ok(gr > 0, f'|grad ro| {gr:.3e}')
ok(gc > 0, f'|grad corr_net| {gc:.3e}')
nb = sum(p.numel() for p in m.base.parameters()); nr = sum(p.numel() for p in [m.ro.weight, m.ro.bias])
print(f'   backbone params {nb} | ro params {nr} | corr_net params {sum(p.numel() for p in m.corr_net.parameters())}')

print('T3 one optimiser step actually changes backbone weights (not just non-zero grad)')
w0 = m.base.encoder.layers[-1].mixer.in_proj.weight.clone()
opt = torch.optim.Adam(list(m.base.parameters()) + [m.ro.weight, m.ro.bias] + list(m.corr_net.parameters()), lr=1e-3)
opt.step()
ok((m.base.encoder.layers[-1].mixer.in_proj.weight - w0).abs().max().item() > 0, 'backbone weight changed after opt.step()')
print('ALL PASS')
