"""PAT-304 R1 gates (Linear 722e5953 / d4ad8aee acceptance): block replication vs the official layers, step vs full scan (forward and
gradients), whole-pipeline equality with a pure step-by-step reference, variable lookback / horizon, VPT permutation handling (per-sample
pi fixed over the window, scan-position readout + pi^-1, equivariance, identity pi == fixed time-major scan, deterministic eval order),
VPT cost bookkeeping, one training step. usage: python patchmem/test_r1.py"""
import sys
from argparse import Namespace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.r1_model import R1

dev = 'cuda'
C, P, D_MODEL = 5, 48, 16
FAILS = []


def ok(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def build(vpt=1, seed=0, e_layers=3):
    torch.manual_seed(seed)
    cfg = Namespace(task_name='long_term_forecast', seq_len=720, pred_len=96, patch_len=P, stride=P, d_model=D_MODEL, e_layers=e_layers,
                    enc_in=C, dropout=0.0, VPT_mode=vpt, ATSP_solver='SA', use_casual_conv=False)
    return R1(cfg).to(dev)


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


torch.manual_seed(1)
x = torch.randn(3, 720, C, device=dev).cumsum(1) * 0.1                          # dataset-scaled lookback, T = 15

print('B  block replication: our fused add+norm + conv-off mixer == the official Block.forward on the same time-major tokens')
m = build().eval()
tok = torch.randn(3, 15 * C, D_MODEL, device=dev)
hid, res = tok, None
ours, resid = tok, None
for i, layer in enumerate(m.enc.layers[:-1]):
    hid, res = layer(hid, res)                                                     # official
    hn, resid = m._norm(layer, ours, resid)
    ours, _ = m._mixer_full(layer.mixer, hn)
    ok(rel(ours, hid) < 1e-5 and rel(resid, res) < 1e-5, f'layer {i}: output rel diff {rel(ours, hid):.1e}, residual {rel(resid, res):.1e}')

print('S  step vs full scan: boundary states and per-layer final states, forward and gradients')
m = build(seed=2).eval()
perm = torch.arange(C, device=dev).expand(3, -1)
tok = m.embed(torch.randn(3, 15, C, P, device=dev)).reshape(3, 15 * C, -1)
bs_full, st_full = m.history(tok)
st = [tok.new_zeros(3, m.enc.layers[i].mixer.d_inner, m.enc.layers[i].mixer.d_state) for i in range(len(m.enc.layers))]
caps = []
for t in range(15):
    st = m.step_time(tok[:, t * C:(t + 1) * C], st)
    caps.append(st[-1])
bs_step = torch.stack(caps, 1)
ok(rel(bs_step, bs_full) < 1e-4, f'boundary states rel diff {rel(bs_step, bs_full):.1e}')
ok(all(rel(a, b) < 1e-4 for a, b in zip(st, st_full)), 'final state of every layer matches ' + ' '.join(f'{rel(a, b):.1e}' for a, b in zip(st, st_full)))
g = {}
for name, fn in (('full', lambda: m.read(m.history(tok)[0])), ('step', None)):
    m.zero_grad()
    if name == 'full':
        out = fn()
    else:
        st = [tok.new_zeros(3, l.mixer.d_inner, l.mixer.d_state) for l in m.enc.layers]
        cs = []
        for t in range(15):
            st = m.step_time(tok[:, t * C:(t + 1) * C], st)
            cs.append(st[-1])
        out = m.read(torch.stack(cs, 1))
    (out.square().mean()).backward(retain_graph=True)
    g[name] = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
worst = max(rel(g['step'][n], g['full'][n]) for n in g['full'])
ok(set(g['full']) == set(g['step']) and worst < 1e-3, f'gradients agree on {len(g["full"])} tensors, worst rel diff {worst:.1e}')

print('R  whole forward (kernel history + boundary readout + autoregressive rollout) == pure step-by-step reference')
m = build(seed=3).eval()
perm = torch.stack([torch.randperm(C, device=dev) for _ in range(3)])
ref_out = m(x, 4, perm=perm)['pred']
means = x.mean(1, keepdim=True); xc = x - means; sd = torch.sqrt(xc.var(1, keepdim=True, unbiased=False) + 1e-5)
xs = m.to_scan((xc / sd).permute(0, 2, 1).reshape(3, C, 15, P).permute(0, 2, 1, 3), perm)
st = [x.new_zeros(3, l.mixer.d_inner, l.mixer.d_state) for l in m.enc.layers]
for t in range(15):
    st = m.step_time(m.embed(xs[:, t]), st)
outs = [m.to_orig(m.read(st[-1]).unsqueeze(1), perm).squeeze(1)]
for _ in range(3):
    st = m.step_time(m.embed(m.to_scan(outs[-1].unsqueeze(1), perm)).squeeze(1), st)
    outs.append(m.to_orig(m.read(st[-1]).unsqueeze(1), perm).squeeze(1))
step_out = (torch.stack(outs, 1).transpose(-1, -2) * sd.unsqueeze(1) + means.unsqueeze(1)).reshape(3, 4 * P, C)
ok(rel(step_out, ref_out) < 1e-4, f'forecast rel diff {rel(step_out, ref_out):.1e}')

print('V  variable lookback / horizon from one model')
for L in (384, 720, 1440):
    for nH in (2, 4, 7):
        xx = torch.randn(2, L, C, device=dev).cumsum(1) * 0.1
        o = m(xx, nH)
        good = tuple(o['pred'].shape) == (2, nH * P, C) and tuple(o['hist'].shape) == (2, L // P - 1, P, C) and torch.isfinite(o['pred']).all()
        ok(bool(good), f'L={L} H={nH * P}: pred {tuple(o["pred"].shape)} hist {tuple(o["hist"].shape)}')

print('P  VPT permutations')
m = build(seed=4).train()
pi = m.scan_order(6, dev)
ok(all(sorted(r.tolist()) == list(range(C)) for r in pi) and len({tuple(r.tolist()) for r in pi}) > 1,
   'training: one valid permutation per sample, samples differ')
tt = m.enc.transition_tuple
ok(tuple(tt.shape) == (6, C, 2) and bool((tt[:, 0, 0] == pi[:, 0]).all() and (tt[:, 0, 1] == pi[:, 0]).all())
   and bool((tt[:, 1:, 0] == pi[:, :-1]).all() and (tt[:, 1:, 1] == pi[:, 1:]).all()),
   'transition tuples = start (pi1, pi1) + within-step pi_k -> pi_k+1 (original channel-major set)')
m.eval()
pi = torch.stack([torch.randperm(C, device=dev) for _ in range(3)])
base = m(x, 3, perm=pi)['pred']
i, j = 0, 3
xsw = x.clone(); xsw[..., [i, j]] = x[..., [j, i]]
pisw = pi.clone(); pisw[pi == i], pisw[pi == j] = j, i
sw = m(xsw, 3, perm=pisw)['pred']
ok((sw[..., [j, i]] - base[..., [i, j]]).abs().max().item() < 1e-5, 'equivariance: swap variables i,j in x and in pi -> predictions of i,j swap')
with torch.no_grad():
    o0 = m(x, 2, perm=pi)['pred']
    m.readout.bias.view(C, P)[0] += 1.0                                            # scan position 0 only
    o1 = m(x, 2, perm=pi)['pred']
    m.readout.bias.view(C, P)[0] -= 1.0
d = (o1 - o0)[:, :P].abs().amax(1)                                                 # (b, C), first future patch
ok(all(d[b, pi[b, 0]] > 0.1 and d[b, [v for v in range(C) if v != pi[b, 0]]].max() < 1e-5 for b in range(3)),
   'readout block k is scan position k: changing block 0 moves only variable pi[0] (restored by pi^-1)')
m_fixed = build(vpt=0, seed=4).eval(); m_fixed.load_state_dict(m.state_dict())
ident = torch.arange(C, device=dev).expand(3, -1)
ok(rel(m_fixed(x, 3)['pred'], m(x, 3, perm=ident)['pred']) < 1e-6, 'identity pi == fixed time-major scan (VPT off)')

print('A  VPT cost bookkeeping and deterministic evaluation order')
m = build(seed=5)
c0 = m.enc.count_matrix.sum().item()
m.train()
for _ in range(20):
    o = m(x, 2)
    m.update_vpt_cost((o['pred'] - x[:, :2 * P]).square().detach())
ok(m.enc.count_matrix.sum().item() - c0 == 20 * 3 * C, f'count matrix grew by {m.enc.count_matrix.sum().item() - c0:.0f} = 20 batches x 3 samples x {C} transitions')
m.eval()
p1 = m(x, 2)['pred']; o1 = list(m.ids_shuffle)
p2 = m(x, 2)['pred']
ok(sorted(o1) == list(range(C)) and m.ids_shuffle == o1 and (p1 - p2).abs().max().item() == 0.0, f'eval order {o1} fixed, forecasts identical')
m.train(); m(x, 2); ok(m.ids_shuffle is None, 'a training forward clears the cached eval order (re-solved at the next eval)')

print('T  one training step: finite, gradients reach embedding, readout and every layer state path')
m = build(seed=6).train()
o = m(x[:, -480:], 3)
loss = (o['pred'] - torch.randn_like(o['pred'])).square().mean() + 0.5 * (o['hist'] - x[:, -480 + P:].reshape(3, -1, P, C)).square().mean()
loss.backward()
gn = lambda mod: sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
ok(torch.isfinite(loss).item() and gn(m.emb) > 0 and gn(m.readout) > 0 and all(gn(l.mixer.in_proj) > 0 for l in m.enc.layers),
   f'loss {loss.item():.4f}; grad |emb| {gn(m.emb):.2e} |readout| {gn(m.readout):.2e}')
ok(gn(m.enc.layers[-1].mixer.out_proj) == 0, 'last layer out_proj gets no gradient (its token output is never read) -- expected')

print('ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS))
sys.exit(1 if FAILS else 0)
