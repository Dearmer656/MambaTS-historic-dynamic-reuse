"""PAT-304 R1 gates (Linear 722e5953 / d4ad8aee acceptance): block replication vs the official layers, step vs full scan (forward and
gradients), whole-pipeline equality with a pure step-by-step reference, variable lookback / horizon, VPT permutation handling (per-sample
pi fixed over the window, scan-position readout + pi^-1, equivariance, identity pi == fixed time-major scan, deterministic eval order),
VPT cost bookkeeping, history -> rollout cache gradients vs a pure step reference (G), one real-config training step. usage: python patchmem/test_r1.py"""
import re, sys
from argparse import Namespace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from patchmem.r1_model import R1, atsp_order
from patchmem.ref_scan import mamba_proj
_mixer_full_fixed = R1._mixer_full

dev = 'cuda'
C, P, D_MODEL = 5, 48, 16
FAILS = []


def ok(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def build(vpt=1, seed=0, e_layers=3, c=C, solver='SA'):
    torch.manual_seed(seed)
    cfg = Namespace(task_name='long_term_forecast', seq_len=720, pred_len=96, patch_len=P, stride=P, d_model=D_MODEL, e_layers=e_layers,
                    enc_in=c, dropout=0.0, VPT_mode=vpt, ATSP_solver=solver, use_casual_conv=False)
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

def step_ref(m, x, n_out, perm):
    """Pure token-by-token reference: every layer (history and rollout) through the explicit one-token step, fully differentiable."""
    b, L, c = x.shape
    means = x.mean(1, keepdim=True).detach(); xc = x - means
    sd = torch.sqrt(xc.var(1, keepdim=True, unbiased=False) + 1e-5).detach()
    xs = m.to_scan((xc / sd).permute(0, 2, 1).reshape(b, c, L // P, P).permute(0, 2, 1, 3), perm)
    st = [x.new_zeros(b, l.mixer.d_inner, l.mixer.d_state) for l in m.enc.layers]
    for t in range(L // P):
        st = m.step_time(m.embed(xs[:, t]), st)
    outs = [m.to_orig(m.read(st[-1]).unsqueeze(1), perm).squeeze(1)]
    for _ in range(n_out - 1):
        st = m.step_time(m.embed(m.to_scan(outs[-1].unsqueeze(1), perm)).squeeze(1), st)
        outs.append(m.to_orig(m.read(st[-1]).unsqueeze(1), perm).squeeze(1))
    return (torch.stack(outs, 1).transpose(-1, -2) * sd.unsqueeze(1) + means.unsqueeze(1)).reshape(b, n_out * P, c)


print('R  whole forward (kernel history + boundary readout + autoregressive rollout) == pure step-by-step reference')
m = build(seed=3).eval()
perm = torch.stack([torch.randperm(C, device=dev) for _ in range(3)])
with torch.no_grad():
    ref_out = m(x, 4, perm=perm)['pred']
    step_out = step_ref(m, x, 4, perm)
ok(rel(step_out, ref_out) < 1e-4, f'forecast rel diff {rel(step_out, ref_out):.1e}')


def _legacy_mixer_full(mx, hn):
    """b637aac behaviour: the kernel's return_last_state (no gradient) as the rollout cache."""
    xx, dt, A, B, Cm, D, z, bias = mamba_proj(mx, hn)
    y, last = selective_scan_fn(xx, dt, A, B, Cm, D, z=z, delta_bias=bias, delta_softplus=True, return_last_state=True)
    return mx.out_proj(y.transpose(1, 2)), last


print('G  history -> rollout cache gradient (loss on future patches 2..n only, which read the lower-layer caches)')
m = build(seed=7, e_layers=4).eval()
perm = torch.stack([torch.randperm(C, device=dev) for _ in range(3)])
tok = m.embed(torch.randn(3, 15, C, P, device=dev)).reshape(3, 15 * C, -1)
hn, resid = m._norm(m.enc.layers[0], tok, None)
xx, dt, A, B, Cm, D, z, bias = mamba_proj(m.enc.layers[0].mixer, hn)
_, kern_last = _legacy_mixer_full(m.enc.layers[0].mixer, hn)
cf = m._final_state(xx, dt, A, B, bias)
ok(rel(cf, kern_last) < 1e-4, f'forward: closed-form final state == kernel last_state, rel diff {rel(cf, kern_last):.1e}')
wts = torch.randn(3, 3 * P, C, device=dev)


def grads(fn):
    m.zero_grad()
    xg = x.clone().requires_grad_(True)
    (fn(xg)[:, P:] * wts).sum().backward()                                          # future patches 2..4 only
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}, xg.grad.detach().clone()


g_ref, gx_ref = grads(lambda xg: step_ref(m, xg, 4, perm))
g_new, gx_new = grads(lambda xg: m(xg, 4, perm=perm)['pred'])
R1._mixer_full = staticmethod(_legacy_mixer_full)
g_old, gx_old = grads(lambda xg: m(xg, 4, perm=perm)['pred'])
R1._mixer_full = staticmethod(_mixer_full_fixed)
lower = [n for n in g_ref if re.match(rf'base\.encoder\.layers\.(\d+)\.mixer\.', n) and int(n.split('.')[3]) < len(m.enc.layers) - 1]
w_new = max(rel(g_new.get(n, torch.zeros_like(g_ref[n])), g_ref[n]) for n in g_ref)
w_old = max(rel(g_old.get(n, torch.zeros_like(g_ref[n])), g_ref[n]) for n in lower)
print('       lower-layer param grad rel diff vs step reference (before fix / after fix):')
for n in lower:
    print(f'         {n:45s} {rel(g_old.get(n, torch.zeros_like(g_ref[n])), g_ref[n]):.1e} / {rel(g_new[n], g_ref[n]):.1e}')
ok(set(g_new) == set(g_ref) and w_new < 1e-3, f'after fix: all {len(g_ref)} param grads match the step reference, worst rel diff {w_new:.1e}')
ok(rel(gx_new, gx_ref) < 1e-3, f'after fix: input grad rel diff {rel(gx_new, gx_ref):.1e} (before fix {rel(gx_old, gx_ref):.1e})')
ok(w_old > 1e-2, f'repro: the b637aac cache (kernel last_state) gives wrong lower-layer grads, worst rel diff {w_old:.1e}')

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

def expected_vpt(tt, loss_bhc, c):
    """Independent per-sample, per-edge reference of the VPT cost update: sample b adds cost_b / std to each of its C transitions."""
    cost = loss_bhc.double().mean((1, 2)); cost = cost / cost.std()
    tot = torch.zeros(c, c, dtype=torch.float64); cnt = torch.zeros(c, c, dtype=torch.float64)
    for b in range(tt.shape[0]):
        for i, j in tt[b].tolist():
            tot[i, j] += cost[b].item(); cnt[i, j] += 1
    return tot, cnt


def fmt(mat, cnt):
    return ' '.join(f'({i},{j}):{mat[i, j]:.3f}' for i in range(mat.shape[0]) for j in range(mat.shape[1]) if cnt[i, j] > 0)


print('A  VPT cost attribution (exact, per sample and per edge), degenerate batches, sparse adjacency, deterministic eval order')
m = build(seed=5, c=3)
tt = torch.tensor([[[0, 0], [0, 1], [1, 2]], [[2, 2], [2, 0], [0, 1]]], device=dev)       # pi = [0,1,2] and [2,0,1]
loss = torch.stack([torch.full((2 * P, 3), 1.0), torch.full((2 * P, 3), 10.0)]).to(dev)   # per-sample cost 1 and 10
tot_ref, cnt_ref = expected_vpt(tt.cpu(), loss.cpu(), 3)
m.enc.transition_tuple = tt
m.update_vpt_cost(loss)
tot_new, cnt_new = m.enc.adjacency_matrix.double().cpu(), m.enc.count_matrix.double().cpu()
mo = build(seed=5, c=3); mo.enc.transition_tuple = tt
mo.enc.batch_update_state(loss)                                                          # original MambaTS method (A0 path)
tot_old = mo.enc.adjacency_matrix.double().cpu()
print(f'       B=2 C=3 hand case, cost [1, 10] -> normalised {(torch.tensor([1., 10.]) / torch.tensor([1., 10.]).std()).tolist()}')
print(f'         expected     {fmt(tot_ref, cnt_ref)}')
print(f'         original     {fmt(tot_old, cnt_ref)}')
print(f'         R1 (fixed)   {fmt(tot_new, cnt_ref)}')
ok(torch.allclose(tot_new, tot_ref, atol=1e-5) and torch.equal(cnt_new, cnt_ref), 'fixed update == per-sample per-edge reference (sum and count)')
ok(not torch.allclose(tot_old, tot_ref, atol=1e-3), 'repro: the original batch_update_state (cost.repeat(C)) misattributes the hand case')
seen = cnt_ref > 0
ok(bool(torch.isnan(mo.enc.get_adjacency_matrix()).any()), 'repro: original get_adjacency_matrix is NaN with unobserved edges')
adj = m.vpt_adjacency().cpu().double()
mean = tot_ref / cnt_ref.clamp_min(1); shift = mean[seen].min().abs() + 1e-7
ok(bool(torch.isfinite(adj).all()) and torch.allclose(adj[seen], mean[seen] + shift, atol=1e-5)
   and torch.allclose(adj[~seen], (mean[seen].max() + shift).expand(int((~seen).sum())), atol=1e-5),
   f'sparse adjacency finite; observed = mean cost, {int((~seen).sum())} unobserved edges = max observed (never preferred)')
for solver in ('SA', 'LS', 'GD'):
    order = atsp_order(m.vpt_adjacency(), solver, 3)
    ok(sorted(int(v) for v in order) == [0, 1, 2], f'ATSP {solver} on the sparse adjacency returns a valid permutation {order}')
before = (m.enc.adjacency_matrix.clone(), m.enc.count_matrix.clone())
m.enc.transition_tuple = tt[:1]; m.update_vpt_cost(loss[:1])                             # B = 1: std undefined
m.enc.transition_tuple = tt; m.update_vpt_cost(torch.ones_like(loss))                    # equal costs: std 0
ok(m.vpt_skipped == 2 and torch.equal(m.enc.adjacency_matrix, before[0]) and torch.equal(m.enc.count_matrix, before[1])
   and bool(torch.isfinite(m.vpt_adjacency()).all()), f'B=1 and zero-variance batches skipped ({m.vpt_skipped}), statistics unchanged and finite')

m = build(seed=5)
m.train()
tot_ref = torch.zeros(C, C, dtype=torch.float64); cnt_ref = torch.zeros(C, C, dtype=torch.float64)
for _ in range(20):
    o = m(x, 2)
    el = (o['pred'] - x[:, :2 * P]).square().detach()
    t_, c_ = expected_vpt(m.enc.transition_tuple.cpu(), el.cpu(), C); tot_ref += t_; cnt_ref += c_
    m.update_vpt_cost(el)
ok(torch.allclose(m.enc.adjacency_matrix.double().cpu(), tot_ref, atol=1e-4) and torch.equal(m.enc.count_matrix.double().cpu(), cnt_ref)
   and cnt_ref.sum().item() == 20 * 3 * C, f'20 random training batches: every edge sum/count == per-sample reference ({cnt_ref.sum().item():.0f} transitions)')
m.eval()
p1 = m(x, 2)['pred']; o1 = list(m.ids_shuffle)
p2 = m(x, 2)['pred']
ok(sorted(o1) == list(range(C)) and m.ids_shuffle == o1 and (p1 - p2).abs().max().item() == 0.0, f'eval order {o1} fixed, forecasts identical')
m.train(); m(x, 2); ok(m.ids_shuffle is None, 'a training forward clears the cached eval order (re-solved at the next eval)')

print('T  one training step, real ETTh2 config (C=7, e_layers=5, dropout=0.3, train mode, lambda_hist=0.5): finite, Adam updates')
torch.manual_seed(6)
cfg = Namespace(task_name='long_term_forecast', seq_len=1440, pred_len=336, patch_len=P, stride=P, d_model=16, e_layers=5,
                enc_in=7, dropout=0.3, VPT_mode=1, ATSP_solver='SA', use_casual_conv=False)
m = R1(cfg).to(dev).train()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
x7 = torch.randn(8, 1440 + 336, 7, device=dev).cumsum(1) * 0.1
o = m(x7[:, :720], 3)
loss = (o['pred'] - x7[:, 720:720 + 3 * P]).square().mean() + 0.5 * (o['hist'] - x7[:, P:720].reshape(8, -1, P, 7)).square().mean()
loss.backward()
gn = lambda mod: sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
fin = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
ok(torch.isfinite(loss).item() and fin and gn(m.emb) > 0 and gn(m.readout) > 0 and all(gn(l.mixer.in_proj) > 0 for l in m.enc.layers),
   f'loss {loss.item():.4f}, all grads finite; grad |emb| {gn(m.emb):.2e} |readout| {gn(m.readout):.2e}')
dead = [n for n, p in m.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
print('       params with no / zero grad (expected dead path: last layer token output, norm_f):\n         ' + '\n         '.join(dead))
nl = len(m.enc.layers) - 1
ok(all(n.startswith(f'base.encoder.layers.{nl}.mixer.') or n.startswith('base.encoder.norm_f.') for n in dead),
   'every dead parameter belongs to the last layer token-output path or norm_f')
before = {n: p.detach().clone() for n, p in m.named_parameters()}
opt.step()
live = [n for n, p in m.named_parameters() if p.requires_grad and n not in dead]
moved = [n for n in live if not torch.equal(dict(m.named_parameters())[n], before[n])]
ok(len(moved) == len(live) and all(torch.isfinite(p).all() for p in m.parameters()), f'Adam step moved all {len(moved)} live params, all finite')
m.update_vpt_cost((o['pred'] - x7[:, 720:720 + 3 * P]).square().detach())

print('ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: ' + '; '.join(FAILS))
sys.exit(1 if FAILS else 0)
