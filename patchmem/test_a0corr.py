"""PAT-304 A0Corr tests (GPU: official Mamba kernels). usage: python patchmem/test_a0corr.py <A0 checkpoint.pth>"""
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


def ok(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg); assert cond, msg


def build(shuffle=False):
    base = Model(cfg).float(); base.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
    return A0Corr(base, order, corr='shared', shuffle=shuffle).to(dev)


order = [6, 2, 1, 5, 0, 4, 3]
torch.manual_seed(0)
x = torch.randn(4, P * NP, C, device=dev).cumsum(1) * 0.1
m = build(); m.train()
with torch.no_grad():
    m.aux.weight.normal_(0, 0.3); m.aux.bias.normal_(0, 0.1)
ref = m.base; ref.encoder.ids_shuffle = torch.tensor(order, device=dev)[None].repeat(x.shape[0], 1)

print('T1 zero-initialised correction == official A0 forward (fixed order, eval)')
with torch.no_grad():
    y0 = ref(x.clone(), None, None, None)
    m.use_corr = False; yf = m(x); m.use_corr = True
    rec = []; o = m.run(x, record=rec)
ok((yf - y0).abs().max().item() < 1e-5, f'fast path max diff {(yf - y0).abs().max().item():.1e}')
ok((o['pred'] - y0).abs().max().item() < 1e-4, f'token-by-token scan max diff {(o["pred"] - y0).abs().max().item():.1e}')
ok(not ref.training and m.training, 'backbone stays in eval mode while A0Corr trains')

print('T2 residual e_s = x_s - aux(o_{s-1}) for every within-variable patch t >= 1 (except the last scan token)')
ss = [s for s, _ in rec]
ok(ss == [s for s in range(C * NP - 1) if s % NP >= 1], f'{len(ss)} correction sites')
with torch.no_grad():
    f, yt = m.aux_pairs(x)
    pred = m.aux(f).reshape(x.shape[0], C, NP - 1, P); yt = yt.reshape(x.shape[0], C, NP - 1, P)
    ref_e = {k * NP + t + 1: (yt[:, k, t] - pred[:, k, t]) for k in range(C) for t in range(NP - 1)}
    err = max((e - ref_e[s]).abs().max().item() for s, e in rec)
ok(err < 1e-4, f'max diff {err:.1e}')

print('T3 nonzero correction changes the forecast; shuffle control differs')
with torch.no_grad():
    m.corr_net.W2.normal_(0, 0.05)
    y1 = m(x)
ok((y1 - y0).abs().max().item() > 1e-4, f'max change {(y1 - y0).abs().max().item():.3f}')
ms = build(shuffle=True); ms.load_state_dict(m.state_dict())
with torch.no_grad():
    ok((ms(x) - y1).abs().max().item() > 1e-5, 'shuffled residual gives a different forecast')

print('T4 gradients: only CorrNet is trained')
m.zero_grad(); out = m.run(x); (out['pred'].square().mean() + 1e-4 * out['reg']).backward()
gsum = lambda ps: sum(0.0 if p.grad is None else p.grad.abs().sum().item() for p in ps)
ok(gsum(m.corr_net.parameters()) > 0, f'|grad CorrNet| {gsum(m.corr_net.parameters()):.2e}')
ok(all(p.grad is None for p in m.base.parameters()) and m.aux.weight.grad is None, 'no gradient on the frozen A0 or aux')
ok(sorted(n for n, p in m.named_parameters() if p.requires_grad) == sorted('corr_net.' + n for n, _ in m.corr_net.named_parameters()), 'trainable = CorrNet')
print('T5 per-step loss: equals the uncorrected next-patch MSE at zero correction, and trains CorrNet')
m2 = build(); m2.load_state_dict(m.state_dict())
with torch.no_grad():
    m2.corr_net.W2.zero_(); m2.corr_net.b2.zero_()
    o0 = m2.run(x); f, yt = m2.aux_pairs(x)
    ref_step = (yt - m2.aux(f)).reshape(x.shape[0], C, NP - 1, P)[:, :, 1:].square().mean().item()   # corrected sites t = 1..13 predict t+1
ok(abs(o0['step'].item() - ref_step) < 1e-4, f"step {o0['step'].item():.5f} vs uncorrected {ref_step:.5f}")
m.zero_grad(); out = m.run(x); out['step'].backward()
ok(gsum(m.corr_net.parameters()) > 0 and all(p.grad is None for p in m.base.parameters()), 'step loss reaches CorrNet only')
print(f'   CorrNet params {sum(p.numel() for p in m.corr_net.parameters())}')
print('ALL PASS')
