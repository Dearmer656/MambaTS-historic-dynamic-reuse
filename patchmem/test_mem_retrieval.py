"""PAT-304 state-level retrieval memory (A0Corr mem=none|real|shuf) gates before any full run (prereg a87e5642 + revision 535b5e5f).
G3 init/identity, G1 gradient credit (>= 2 distinct historical slots, per-loss, path a = retrieval vs path b = carry), G2 causality on a
real Weather window, plus the M2 shuffle sanity check. usage: python patchmem/test_mem_retrieval.py"""
import sys
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.MambaTS import Model
from patchmem.a0corr import A0Corr

dev = 'cuda'
P, NP, H = 48, 15, 96
FAILS = []


def ok(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def cfg_for(C):
    return Namespace(task_name='long_term_forecast', seq_len=P * NP, pred_len=H, patch_len=P, stride=P, d_model=16, e_layers=2, enc_in=C,
                     dropout=0.0, VPT_mode=1, ATSP_solver='SA', use_casual_conv=False)


def build(mem, C, live_corr=True, open_gate=True, seed=0):
    torch.manual_seed(seed)
    cfg = cfg_for(C)
    base = Model(cfg).float().to(dev); base.encoder.eval()
    with torch.no_grad():
        base.encoder(torch.zeros(1, cfg.enc_in * (cfg.seq_len // cfg.patch_len), cfg.d_model, device=dev))
    order = base.encoder.ids_shuffle[0].tolist()
    m = A0Corr(base, order, corr='gated', mode='cross', freeze=False, mem=mem).to(dev)
    g = torch.Generator(device='cpu').manual_seed(seed + 1)
    with torch.no_grad():
        m.ro.weight.copy_(torch.randn(m.ro.weight.shape, generator=g) * 0.02)
        if live_corr:                                                    # Wc = 0 at default init -> C = 0 -> m = 0; tests need non-trivial values
            m.corr_net.Wc.copy_(torch.randn(m.corr_net.Wc.shape, generator=g) * 0.05)
        if open_gate and m.mem_net is not None:
            m.mem_net.Wg.bias.fill_(0.0)                                 # g = 0.5 so the memory visibly matters
    return m.eval()


def gsum(ps):
    return sum(p.grad.abs().sum().item() for p in ps if p.grad is not None)


def grads(m, loss, Cs):
    m.zero_grad()
    for c in Cs:
        c.grad = None
    loss.backward(retain_graph=True)
    mn = m.mem_net
    return dict(Wq=gsum([mn.Wq.weight]), Wk=gsum([mn.Wk.weight]), tau=gsum([mn.log_tau]), Wg=gsum([mn.Wg.weight, mn.Wg.bias]),
                corr=gsum(m.corr_net.parameters()), C=[(c.grad.abs().sum().item() if c.grad is not None else 0.0) for c in Cs])


C5 = 5
x = torch.randn(2, P * NP, C5, device=dev).cumsum(1) * 0.1
l1 = NP - 1                                                              # corrected sites per variable (t = 0..13)

print('G3a memory gated to exactly zero reproduces M0 (mem=none) bit for bit: pred, step, step_mem')
m0 = build('none', C5); m1 = build('real', C5)
missing = m1.load_state_dict(m0.state_dict(), strict=False)
ok(all(k.startswith('mem_net.') for k in missing.missing_keys) and not missing.unexpected_keys, f'only mem_net params differ (missing {len(missing.missing_keys)})')
with torch.no_grad():
    o0 = m0.run_cross(x); m1.mem_net.audit = 'zero'; o1 = m1.run_cross(x); m1.mem_net.audit = None
d = max((o0['pred'] - o1['pred']).abs().max().item(), abs(o0['step'].item() - o1['step'].item()), abs(o0['step_mem'].item() - o1['step_mem'].item()))
ok(d == 0.0, f'max |diff| {d:.2e}')

print('G3b default init (b_g = -4, W_g = 0): gate ~ sigmoid(-4) everywhere, close to but not identical with M0')
m1d = build('real', C5, open_gate=False); m1d.load_state_dict(m0.state_dict(), strict=False)
with torch.no_grad():
    od = m1d.run_cross(x)
rel = ((od['pred'] - o0['pred']).norm() / o0['pred'].norm()).item()
ok(abs(od['mem_stats']['g'] - torch.sigmoid(torch.tensor(-4.0)).item()) < 1e-6, f"mean gate {od['mem_stats']['g']:.5f}")
ok(0 < rel < 0.05, f'relative pred change vs M0 {rel:.2e}')

print('G3c default init with Wc = 0 (C = 0, so m = 0): no gradient to W_Q / W_K / W_g yet -- expected, recorded, not a failure')
mz = build('real', C5, live_corr=False, open_gate=False)
mz.train(); oz = mz.run_cross(x); mz.zero_grad(); (oz['pred'].square().mean() + oz['step_mem']).backward()
print(f"   |grad| Wq {gsum([mz.mem_net.Wq.weight]):.2e} Wk {gsum([mz.mem_net.Wk.weight]):.2e} Wg {gsum([mz.mem_net.Wg.weight]):.2e} | corr_net {gsum(mz.corr_net.parameters()):.2e} (CorrNet still learns from the start)")

print('G1 gradient credit: target site t = 3 of scan position k = 0 has M = 3 distinct historical slots (t\' = 0, 1, 2)')
k0, t0 = 0, 3
idx = k0 * l1 + t0
for loss_name in ('step_mem_t', 'mse_H96'):
    res = {}
    for surgery in ('full', 'cut_carry', 'cut_value', 'cut_both'):
        m = build('real', C5); m.train()
        m.cut_value = surgery in ('cut_value', 'cut_both'); m.cut_carry = surgery in ('cut_carry', 'cut_both')
        tr = {}
        out = m.run_cross(x, trace=tr)
        Cs = tr['C'][k0 * l1: k0 * l1 + l1]                              # this variable's corrections, t = 0..13
        for c in Cs:
            c.retain_grad()
        loss = out['steps_mem'][idx] if loss_name == 'step_mem_t' else out['pred'].square().mean()
        res[surgery] = grads(m, loss, Cs)
    f = res['full']; ca = res['cut_carry']; cv = res['cut_value']; cb = res['cut_both']
    print(f'  [{loss_name}] |grad| full: Wq {f["Wq"]:.2e} Wk {f["Wk"]:.2e} tau {f["tau"]:.2e} Wg {f["Wg"]:.2e} corr_net {f["corr"]:.2e}')
    print(f'  [{loss_name}] CorrNet |grad| by path: both {f["corr"]:.3e} | retrieval only (cut_carry) {ca["corr"]:.3e} | carry only (cut_value) {cv["corr"]:.3e} | none {cb["corr"]:.3e}')
    print(f'  [{loss_name}] |grad C_j| j = 0..5, retrieval only: ' + ' '.join(f'{v:.2e}' for v in ca['C'][:6]))
    fin = all(np.isfinite(v) for v in (f['Wq'], f['Wk'], f['tau'], f['Wg'], f['corr']))
    ok(fin and min(f['Wq'], f['Wk'], f['tau'], f['Wg']) > 0, f'{loss_name}: W_Q, W_K, tau, W_g get finite non-zero gradient')
    ok(ca['corr'] > 0 and all(v > 0 for v in ca['C'][:t0]), f'{loss_name}: retrieval path alone (carry cut) reaches CorrNet and every C_j, j < t')
    if loss_name == 'step_mem_t':
        ok(all(v == 0 for v in ca['C'][t0:]), f'{loss_name}: no credit to C_j, j >= t (written at or after the read)')
    ok(cb['corr'] == 0 and all(v == 0 for v in cb['C']), f'{loss_name}: both paths cut -> zero gradient to CorrNet (paths fully accounted)')

print('M2 shuffle: values rolled by one slot along time within the sequence (independent of batch size); identical to M1 at t = 1 (M = 1)')
m1 = build('real', C5); m2 = build('shuf', C5); m2.load_state_dict(m1.state_dict())
xb1 = x[:1]                                                              # batch size 1, the official test loader setting
with torch.no_grad():
    t1, t2 = {}, {}; m1.run_cross(xb1, trace=t1); m2.run_cross(xb1, trace=t2)
dy = [(a['yhat'] - b['yhat']).abs().max().item() for a, b in zip(t1['mem'], t2['mem'])]
ok(dy[1] < 1e-6 and max(dy[2:l1]) > 1e-4, f'bs=1: |yhat_mem M1 - M2| at t=1 {dy[1]:.1e}, max over t>=2 {max(dy[2:l1]):.1e}')

print('G2 causality on a real Weather window (21 variables, 720 steps); site (k*, t*) = (scan position 2, t = 5)')
df = pd.read_csv(ROOT / 'dataset/weather/weather.csv')
raw = df.iloc[:, 1:].values.astype(np.float32)[:P * NP]
raw = (raw - raw.mean(0)) / (raw.std(0) + 1e-5)
xw = torch.as_tensor(raw, device=dev)[None]
mw = build('real', 21); order = mw.order.tolist(); ks, ts = 2, 5; v = order[ks]
site = ks * l1 + ts


def site_view(m, xin, freeze_norm_from=None):
    """Memory quantities at site (ks, ts): a, injection, keys, yhat_mem, and this variable's corrections C_j, j <= ts."""
    if freeze_norm_from is not None:                                     # reuse the clean window's RevIN statistics
        _, mu, sd = m.norm(freeze_norm_from)
        m.norm = lambda xe: ((xe - mu) / sd, mu, sd)                     # instance attribute shadows the method
    tr = {}
    with torch.no_grad():
        m.run_cross(xin, trace=tr)
    if freeze_norm_from is not None:
        del m.norm
    rec = tr['mem'][site]
    return [rec['a'], rec['inj'], rec['keys'], rec['yhat']] + tr['C'][ks * l1: ks * l1 + ts + 1]


def maxdiff(A, B):
    return max((a - b).abs().max().item() for a, b in zip(A, B))


ref = site_view(mw, xw)
gen = torch.Generator(device='cpu').manual_seed(7)
xa = xw.clone(); xa[0, (ts + 1) * P:, v] += torch.randn(P * NP - (ts + 1) * P, generator=gen).to(dev) * 3
later = order[ks + 1:]
xb = xw.clone(); xb[0, :, later] += torch.randn(P * NP, len(later), generator=gen).to(dev) * 3
print(f'  (i)  perturb own variable x_(k*, t*+1:) [the per-site target and after], RevIN stats from the clean window: max |diff| {maxdiff(ref, site_view(mw, xa, xw)):.2e}')
ok(maxdiff(ref, site_view(mw, xa, xw)) < 1e-5, '(i) q/K/C_j/a/m/yhat_mem at the site unchanged')
dr = maxdiff(ref, site_view(mw, xa))
print(f'       same perturbation, official RevIN on the perturbed window: max |diff| {dr:.2e} -- window statistics see the whole lookback window '
      f'(window-observed protocol; not an online per-step claim)')
ok(maxdiff(ref, site_view(mw, xb)) < 1e-5, f'(ii) perturb all patches of the {len(later)} variables scanned after k*: unchanged, max |diff| {maxdiff(ref, site_view(mw, xb)):.2e}')
print('  (iii) the forecast target Y never enters run_cross (signature takes x_enc only): unchanged by construction')

print('ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS))
sys.exit(1 if FAILS else 0)
