"""PAT-304 A0Corr(step_sg=True) test: the per-step loss reaches CorrNet and ro but never the backbone. usage: python patchmem/test_step_sg.py"""
import sys
from argparse import Namespace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.MambaTS import Model
from patchmem.a0corr import A0Corr

dev = 'cuda'
C, P, NP, H = 5, 48, 15, 96
cfg = Namespace(task_name='long_term_forecast', seq_len=P * NP, pred_len=H, patch_len=P, stride=P, d_model=16, e_layers=2, enc_in=C, dropout=0.0,
                VPT_mode=1, ATSP_solver='SA', use_casual_conv=False)


def ok(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg); assert cond, msg


def build(step_sg):
    torch.manual_seed(0)
    base = Model(cfg).float().to(dev); base.encoder.eval()
    with torch.no_grad():
        base.encoder(torch.zeros(1, cfg.enc_in * (cfg.seq_len // cfg.patch_len), cfg.d_model, device=dev))
    order = base.encoder.ids_shuffle[0].tolist()
    m = A0Corr(base, order, corr='gated', mode='cross', freeze=False, step_sg=step_sg).to(dev)
    with torch.no_grad():
        m.corr_net.Wc.normal_(0, 0.05); m.ro.weight.normal_(0, 0.02)
    return m


x = torch.randn(3, cfg.seq_len, C, device=dev).cumsum(1) * 0.1
gsum = lambda ps: sum(p.grad.abs().sum().item() for p in ps if p.grad is not None)

print('T1 step loss value is identical with and without step_sg (only the gradient path changes)')
m0, m1 = build(False), build(True)
m0.train(); m1.train()
o0, o1 = m0.run_cross(x), m1.run_cross(x)
ok(abs(o0['step'].item() - o1['step'].item()) < 1e-6 and (o0['pred'] - o1['pred']).abs().max().item() < 1e-6,
   f"step {o0['step'].item():.6f} vs {o1['step'].item():.6f}")

print('T2 step_sg=False: step loss reaches the backbone; step_sg=True: it does not, but still reaches CorrNet and ro')
for name, m, o in (('sg off', m0, o0), ('sg on', m1, o1)):
    m.zero_grad(); o['step'].backward()
    gb, gc, gr = gsum(m.base.parameters()), gsum(m.corr_net.parameters()), gsum([m.ro.weight, m.ro.bias])
    print(f'   {name}: |grad backbone| {gb:.3e} | corr_net {gc:.3e} | ro {gr:.3e}')
    if name == 'sg off':
        ok(gb > 0, 'backbone receives step gradient without sg')
    else:
        ok(gb == 0.0 and gc > 0 and gr > 0, 'backbone gets none; CorrNet and ro still trained by the step loss')

print('T3 step_sg=True: the main forecast loss still trains the backbone')
m1.zero_grad(); m1.run_cross(x)['pred'].square().mean().backward()
ok(gsum(m1.base.parameters()) > 0, f'|grad backbone| from mse {gsum(m1.base.parameters()):.3e}')
print('ALL PASS')
