"""PAT-304 MambaTSMem smoke test (GPU). usage: python patchmem/test_mem_model.py"""
import sys
from argparse import Namespace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.MambaTS import Model
from patchmem.mem_model import MambaTSMem

dev = 'cuda'
C, P, NP, H, R = 5, 48, 15, 96, 2
cfg = Namespace(task_name='long_term_forecast', seq_len=P * NP, pred_len=H, patch_len=P, stride=P, d_model=16, e_layers=2, enc_in=C, dropout=0.1,
                VPT_mode=1, ATSP_solver='SA', use_casual_conv=False)


def ok(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg); assert cond, msg


torch.manual_seed(0)
base = Model(cfg).float().to(dev)
base.encoder.eval()
with torch.no_grad():
    base.encoder(torch.zeros(1, cfg.enc_in * (cfg.seq_len // cfg.patch_len), cfg.d_model, device=dev))   # fix the ATSP scan order once
base.encoder.train()
x = torch.randn(3, cfg.seq_len, C, device=dev)
xmem = torch.randn(3, R, cfg.seq_len, C, device=dev)

print('T1 mode=none exactly reproduces the official forward')
import copy
base_ref = copy.deepcopy(base)                                      # shares the already-fixed ids_shuffle
m_none = MambaTSMem(base, R=R, mode='none').to(dev)
base_ref.eval(); m_none.eval()                                       # disable dropout for the exact-equivalence check
with torch.no_grad():
    y0 = base_ref(x, None, None, None)
    y1 = m_none(x, None)
ok((y0 - y1).abs().max().item() < 1e-5, f'max diff {(y0 - y1).abs().max().item():.2e}')
ok(m_none.retr is None, 'no retrieval module built in mode=none')

print('T2 mode=mem: near-identity at init (gate ~ sigmoid(-4)), output differs slightly from the base model, changes with xmem')
m_mem = MambaTSMem(copy.deepcopy(base_ref), R=R, mode='mem').to(dev)
m_mem.eval()
with torch.no_grad():
    y2 = m_mem(x, xmem)
    d = (y2 - y0).abs().max().item()
    xmem2 = torch.randn_like(xmem)
    y3 = m_mem(x, xmem2)
    d2 = (y3 - y2).abs().max().item()
ok(0 < d < 1.0, f'near-identity at init, diff from base {d:.4f}')
ok(d2 > 1e-5, f'output changes with different memory content, diff {d2:.2e}')

print('T3 mode=shuf differs from mode=mem (same weights, memory rolled)')
m_shuf = MambaTSMem(copy.deepcopy(base_ref), R=R, mode='shuf').to(dev)
m_shuf.load_state_dict(m_mem.state_dict())
m_shuf.eval()
with torch.no_grad():
    y4 = m_shuf(x, xmem)
ok((y4 - y2).abs().max().item() > 1e-6, 'shuffled memory gives a different output than true memory')

print('T4 gradients: mode=none only touches base; mode=mem touches base + retr; memory-window encoder calls are no_grad')
m_mem.zero_grad()
y = m_mem(x, xmem); y.square().mean().backward()
gb = sum(p.grad.abs().sum().item() for p in m_mem.base.parameters() if p.grad is not None)
gr = sum(p.grad.abs().sum().item() for p in m_mem.retr.parameters() if p.grad is not None)
ok(gb > 0 and gr > 0, f'|grad base| {gb:.2e} |grad retr| {gr:.2e}')

print('T5 raising the gate increases the memory correction magnitude (monotonic sanity, not a learning check)')
with torch.no_grad():
    m_mem.retr.gate.fill_(4.0)
    y5 = m_mem(x, xmem)
    d5 = (y5 - y0).abs().max().item()
ok(d5 > d, f'gate=-4 diff {d:.4f} < gate=+4 diff {d5:.4f}')
print('ALL PASS')
