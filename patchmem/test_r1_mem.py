"""PAT-304 R1Mem gates: the memory design h+ = h- + dh_mem (C stored as the value only, never carried; C on the memory-corrected state;
retrieval over j < t; rollout retrieves without writing), the training variants B1 (Pure-B) / B2 (Hybrid) / B3 (retrieval off), the
writer loss routing, the inference audits, the diagnostics and the per-load parameter snapshot (comment e18ad84b).
usage: python patchmem/test_r1_mem.py"""
import glob, json, os, random, shutil, sys, tempfile
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.r1_model import R1
from patchmem.r1_mem import R1Mem
from patchmem.param_snapshot import log_model_parameter_stats
from patchmem.exp_r1 import solve_eval_order

dev = 'cuda'
C, P, D_MODEL = 5, 48, 16
FAILS = []


def ok(cond, msg):
    print(('  ok   ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def cfg(c=C, e_layers=3):
    return Namespace(task_name='long_term_forecast', seq_len=720, pred_len=96, patch_len=P, stride=P, d_model=D_MODEL, e_layers=e_layers,
                     enc_in=c, dropout=0.0, VPT_mode=1, ATSP_solver='SA', use_casual_conv=False)


def pair(seed=0, readout_ln=False, **kw):
    """R1 and R1Mem with identical backbone / readout weights."""
    torch.manual_seed(seed)
    r1 = R1(cfg(), readout_ln=readout_ln).to(dev).eval()
    torch.manual_seed(seed + 100)
    rm = R1Mem(cfg(), readout_ln=readout_ln, **kw).to(dev).eval()
    missing, unexpected = rm.load_state_dict(r1.state_dict(), strict=False)
    assert not unexpected and all(k.startswith(('corr_net.', 'mem_net.')) for k in missing), (missing, unexpected)
    return r1, rm


def wake(rm, gate_bias=2.0, seed=7):
    """Non-trivial memory: random CorrNet output layer (C != 0) and an open gate."""
    torch.manual_seed(seed)
    with torch.no_grad():
        if hasattr(rm.corr_net, 'Wc'):                                          # GatedCorrNet (default): content z @ Wc, zero at init
            rm.corr_net.Wc.normal_(0, 0.3)
        else:
            rm.corr_net.W2.normal_(0, 0.3); rm.corr_net.b2.normal_(0, 0.1)
        rm.mem_net.Wg.bias.fill_(gate_bias)
    return rm


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


torch.manual_seed(1)
x = torch.randn(3, 720, C, device=dev).cumsum(1) * 0.1                          # T = 15
perm = torch.stack([torch.randperm(C, device=dev) for _ in range(3)])
N_OUT = 4

print('I  at init (CorrNet output layer zero => C == 0) R1Mem == R1 exactly, pred and hist')
r1, rm = pair()
o1, om = r1(x, N_OUT, perm=perm), rm(x, N_OUT, perm=perm)
ok(rel(om['pred'], o1['pred']) < 1e-5 and rel(om['hist'], o1['hist']) < 1e-5,
   f'pred rel diff {rel(om["pred"], o1["pred"]):.1e}, hist {rel(om["hist"], o1["hist"]):.1e}')

print('N  C is never carried: C != 0 but gate closed (audit zero) => R1Mem == R1 exactly')
r1, rm = pair()
wake(rm)
rm.mem_net.audit = 'zero'
o1, om = r1(x, N_OUT, perm=perm), rm(x, N_OUT, perm=perm)
ok(rm.stats()['C'] > 1e-4, f'C is non-zero (mean C^2 {rm.stats()["C"]:.2e})')
ok(rel(om['pred'], o1['pred']) < 1e-5 and rel(om['hist'], o1['hist']) < 1e-5,
   f'pred rel diff {rel(om["pred"], o1["pred"]):.1e}, hist {rel(om["hist"], o1["hist"]):.1e} (C enters nothing but the memory values)')
rm.mem_net.audit = None

print('M  memory active (C != 0, gate open) changes the forecast, history and rollout')
om = rm(x, N_OUT, perm=perm)
ok(rel(om['pred'], o1['pred']) > 1e-3 and rel(om['hist'], o1['hist']) > 1e-3,
   f'pred rel diff {rel(om["pred"], o1["pred"]):.1e}, hist {rel(om["hist"], o1["hist"]):.1e}')

print('K  causality: the prediction of patch t+1 (hist[:, t]) is unchanged when lookback patches > t change')
T = 720 // P
for t in (2, 6, 11):
    x2 = x.clone()                                                              # RevIN uses the whole lookback: keep its mean / var by
    x2[:, (t + 1) * P:] = x[:, (t + 1) * P:].flip(1)                            # reordering the later values instead of changing them
    h1, h2 = rm(x, N_OUT, perm=perm)['hist'], rm(x2, N_OUT, perm=perm)['hist']
    ok(rel(h2[:, :t + 1], h1[:, :t + 1]) < 1e-5 and rel(h2[:, t + 1:], h1[:, t + 1:]) > 1e-4,
       f't={t}: hist[:, :{t + 1}] rel diff {rel(h2[:, :t + 1], h1[:, :t + 1]):.1e} (must be 0), later {rel(h2[:, t + 1:], h1[:, t + 1:]):.1e}')

print('R  retrieval is over j < t: the residual of patch t (slot t) cannot reach h+_t, only h+_{t+1} onwards')
# reordering patch t changes e_t and so slot t; read(h+_t) must not see it. h-_t itself also depends on patch t, so the last layer is
# made input independent (B = 0: no input write; dt_proj weight = 0: constant decay), leaving the memory as the only input path
r1, rm = pair(seed=3)
wake(rm)
with torch.no_grad():
    lm = rm.enc.layers[-1].mixer
    lm.x_proj.weight[lm.dt_rank:lm.dt_rank + lm.d_state].zero_()               # B = 0
    lm.dt_proj.weight.zero_()                                                   # dt = softplus(bias), input independent
for t in (3, 8):
    x2 = x.clone()
    seg = x2[:, t * P:(t + 1) * P].clone()
    x2[:, t * P:(t + 1) * P] = seg.flip(1)                                       # patch t reordered: e_t (slot t) changes
    h1, h2 = rm(x, N_OUT, perm=perm)['hist'], rm(x2, N_OUT, perm=perm)['hist']
    ok(rel(h2[:, :t + 1], h1[:, :t + 1]) < 1e-5 and rel(h2[:, t + 1:], h1[:, t + 1:]) > 1e-4,
       f't={t}: read(h+_s) for s <= t unchanged ({rel(h2[:, :t + 1], h1[:, :t + 1]):.1e}), s > t changed ({rel(h2[:, t + 1:], h1[:, t + 1:]):.1e})')

print('W  rollout retrieves but never writes: the slot bank after history has T-1 slots, the rollout does not add any')
r1, rm = pair()
wake(rm)
n_ret = {}
orig = rm._retrieve
def spy(h, keys, vals, *a, **k):
    n_ret.setdefault(len(vals), 0); n_ret[len(vals)] += 1
    return orig(h, keys, vals, *a, **k)
rm._retrieve = spy
rm(x, N_OUT, perm=perm)
ok(sorted(n_ret) == list(range(1, T)) and all(n_ret[k] == 1 for k in range(1, T - 1)) and n_ret[T - 1] == N_OUT - 1,
   f'retrievals by bank size {dict(sorted(n_ret.items()))}: history boundary t=2..{T - 1} sees t-1 slots once each, '
   f'the {N_OUT - 1} rollout steps all see the full {T - 1}')
rm._retrieve = orig

print('G  gradients: CorrNet and StateMem are trained through the memory path only (future loss, lambda_hist = 0)')
r1, rm = pair()
wake(rm, gate_bias=0.0)
rm.train()
rm.zero_grad()
out = rm(x, N_OUT, perm=perm)
out['pred'].square().mean().backward()
gc = sum(p.grad.abs().sum().item() for p in rm.corr_net.parameters() if p.grad is not None)
gm = {n: p.grad.abs().sum().item() for n, p in rm.mem_net.named_parameters() if p.grad is not None}
ok(gc > 0, f'CorrNet |grad| {gc:.2e}')
ok(all(v > 0 for v in gm.values()) and set(gm) == {n for n, _ in rm.mem_net.named_parameters()}, f'StateMem grads {gm}')
ok(all(torch.isfinite(p.grad).all() for p in rm.parameters() if p.grad is not None), 'all grads finite')

print('S  corr_shuffle control: residuals rolled across the batch change the memory, not the memory-free path')
r1, rm = pair()
wake(rm)
o_a = rm(x, N_OUT, perm=perm)['pred']
rm.corr_shuffle = True
o_b = rm(x, N_OUT, perm=perm)['pred']
ok(rel(o_b, o_a) > 1e-4, f'shuffled residual changes the forecast ({rel(o_b, o_a):.1e})')

print('A  writer loss L_corr = ||R_sg(sg h+ + C) - x_(t+1)||^2: forecast unchanged, trains CorrNet only, training only')
r1, rm = pair()
wake(rm, gate_bias=0.0)
rm.train()
p_off = rm(x, N_OUT, perm=perm)
rm.corr_aux = True
p_on = rm(x, N_OUT, perm=perm)
ok(rel(p_on['pred'], p_off['pred']) == 0 and rel(p_on['hist'], p_off['hist']) == 0, 'pred / hist identical with the writer loss on (h+ + C is never read nor carried)')
ok(rm.corr_loss is not None and torch.isfinite(rm.corr_loss), f'L_corr finite ({rm.corr_loss.item():.4f})')
rm.zero_grad()
rm.corr_loss.backward()
named = dict(rm.named_parameters())
g = {n: (p.grad.abs().sum().item() if p.grad is not None else 0.0) for n, p in named.items()}
ok(sum(v for n, v in g.items() if n.startswith('corr_net.')) > 0, 'CorrNet receives gradient from L_corr')
leak = {n: v for n, v in g.items() if not n.startswith('corr_net.') and v > 0}
ok(not leak, f'no gradient outside CorrNet (readout / backbone / StateMem): {sorted(leak)[:5]}')
rm.eval()
rm(x, N_OUT, perm=perm)
ok(rm.corr_loss is None, 'L_corr not computed in eval mode')
rm.corr_aux = False

def input_free_last_layer(rm):
    """Last layer made input independent (B = 0, constant dt): the memory is then its only input path."""
    with torch.no_grad():
        lm = rm.enc.layers[-1].mixer
        lm.x_proj.weight[lm.dt_rank:lm.dt_rank + lm.d_state].zero_()
        lm.dt_proj.weight.zero_()


def grads_by_prefix(m):
    out = {}
    for n, p in m.named_parameters():
        k = n.split('.')[0] if n.split('.')[0] != 'base' else 'backbone'
        out[k] = out.get(k, 0.0) + (p.grad.abs().sum().item() if p.grad is not None else 0.0)
    return out


print('Y  designed delay: the last history slot C_{T-1} does not reach the first future patch, only the later rollout patches')
r1, rm = pair(seed=3)
wake(rm)
input_free_last_layer(rm)
x2 = x.clone()
x2[:, (T - 1) * P:] = x[:, (T - 1) * P:].flip(1)                                 # last patch reordered: only e_{T-1} / slot T-1 change
p1, p2 = rm(x, N_OUT, perm=perm)['pred'], rm(x2, N_OUT, perm=perm)['pred']
ok(rel(p2[:, :P], p1[:, :P]) < 1e-5 and rel(p2[:, P:], p1[:, P:]) > 1e-4,
   f'first future patch unchanged ({rel(p2[:, :P], p1[:, :P]):.1e}), later rollout patches changed ({rel(p2[:, P:], p1[:, P:]):.1e})')

print('B  B2 Hybrid vs B1 Pure-B: L_future ALONE reaches CorrNet through the memory values only in Hybrid')
for detach in (False, True):
    r1, rm = pair(value_detach=detach)
    wake(rm, gate_bias=0.0)
    rm.train(); rm.zero_grad()
    rm(x, N_OUT, perm=perm)['pred'].square().mean().backward()
    gp = grads_by_prefix(rm)
    tag = 'B1 Pure-B (values detached)' if detach else 'B2 Hybrid'
    ok((gp['corr_net'] == 0) if detach else (gp['corr_net'] > 0), f'{tag}: CorrNet grad from L_future {gp["corr_net"]:.2e} '
       f'({"must be 0" if detach else "must be > 0"})')
    ok(gp['mem_net'] > 0 and gp['readout'] > 0 and gp['backbone'] > 0,
       f'{tag}: StateMem {gp["mem_net"]:.2e}, readout {gp["readout"]:.2e}, backbone {gp["backbone"]:.2e} all > 0')
    rm.corr_aux = True; rm.zero_grad()
    rm(x, N_OUT, perm=perm); rm.corr_loss.backward()
    gp = grads_by_prefix(rm)
    ok(gp['corr_net'] > 0 and all(v == 0 for k, v in gp.items() if k != 'corr_net'), f'{tag}: L_corr ALONE reaches CorrNet only {gp}')

print('X  B3 (retrieve=False): C != 0 and gate open but no injection => forecast == R1 exactly; the writer loss still trains CorrNet')
r1, rm = pair(retrieve=False)
wake(rm)
o1, om = r1(x, N_OUT, perm=perm), rm(x, N_OUT, perm=perm)
ok(rel(om['pred'], o1['pred']) < 1e-5 and rel(om['hist'], o1['hist']) < 1e-5 and rm.stats()['C'] > 1e-4,
   f'pred {rel(om["pred"], o1["pred"]):.1e} hist {rel(om["hist"], o1["hist"]):.1e} with mean C^2 {rm.stats()["C"]:.2e}')
rm.train(); rm.corr_aux = True; rm.zero_grad()
rm(x, N_OUT, perm=perm); rm.corr_loss.backward()
gp = grads_by_prefix(rm)
ok(gp['corr_net'] > 0 and gp['mem_net'] == 0, f'B3: CorrNet grad {gp["corr_net"]:.2e} > 0, StateMem {gp["mem_net"]:.2e} == 0')

print('U  inference audits on one model: zero == R1; uniform and perm change the forecast; perm deterministic per seed')
r1, rm = pair()
wake(rm)
o1 = r1(x, N_OUT, perm=perm)['pred']
outs = {}
for mode, seed in ((None, 0), ('zero', 0), ('uniform', 0), ('perm', 0), ('perm', 0), ('perm', 1)):
    rm.set_audit(mode, seed)
    outs.setdefault((mode, seed), []).append(rm(x, N_OUT, perm=perm)['pred'])
rm.set_audit(None)
ok(rel(outs[('zero', 0)][0], o1) < 1e-5, f'zero == R1 ({rel(outs[("zero", 0)][0], o1):.1e})')
ok(rel(outs[('uniform', 0)][0], outs[(None, 0)][0]) > 1e-4, f'uniform != normal ({rel(outs[("uniform", 0)][0], outs[(None, 0)][0]):.1e})')
ok(rel(outs[('perm', 0)][0], outs[(None, 0)][0]) > 1e-4, f'perm != normal ({rel(outs[("perm", 0)][0], outs[(None, 0)][0]):.1e})')
ok(rel(outs[('perm', 0)][1], outs[('perm', 0)][0]) == 0 and rel(outs[('perm', 1)][0], outs[('perm', 0)][0]) > 1e-6,
   'perm: same seed reproduces exactly, another seed differs')
ok(rm.audit is None and rm.mem_net.audit is None and rel(rm(x, N_OUT, perm=perm)['pred'], outs[(None, 0)][0]) == 0, 'audit reset restores normal')

print('H  B2-H (roll_retrieve=False): history and the first future patch == B2; no rollout retrieval; later patches differ')
r1, rm = pair()
wake(rm)
_, rh = pair(roll_retrieve=False)
rh.load_state_dict(rm.state_dict())
ob, oh = rm(x, N_OUT, perm=perm), rh(x, N_OUT, perm=perm)
ok(rel(oh['hist'], ob['hist']) < 1e-6 and rel(oh['pred'][:, :P], ob['pred'][:, :P]) < 1e-6 and rel(oh['pred'][:, P:], ob['pred'][:, P:]) > 1e-4,
   f'hist {rel(oh["hist"], ob["hist"]):.1e}, first patch {rel(oh["pred"][:, :P], ob["pred"][:, :P]):.1e}, later {rel(oh["pred"][:, P:], ob["pred"][:, P:]):.1e}')
n_ret = {}
orig = rh._retrieve
rh._retrieve = spy
rh(x, N_OUT, perm=perm)
rh._retrieve = orig
ok(sorted(n_ret) == list(range(1, T - 1)) and all(v == 1 for v in n_ret.values()),
   f'retrievals by bank size {dict(sorted(n_ret.items()))}: history boundaries only, none with the full {T - 1}-slot bank')

print('D  B2-drop (slot_keep): evaluation uses every slot (== B2); training with keep 0 == R1; keep 0.5 masks dropped slots per sample')
r1, rm = pair()
wake(rm)
_, rd = pair(slot_keep=0.5)
rd.load_state_dict(rm.state_dict())
ok(rel(rd(x, N_OUT, perm=perm)['pred'], rm(x, N_OUT, perm=perm)['pred']) == 0, 'eval: slot_keep 0.5 == slot_keep 1 exactly')
rd.slot_keep = 0.0
rd.train(); r1.train()                                                          # cfg dropout is 0: train mode is deterministic
o1, od = r1(x, N_OUT, perm=perm), rd(x, N_OUT, perm=perm)
ok(rel(od['pred'], o1['pred']) < 1e-5 and rel(od['hist'], o1['hist']) < 1e-5,
   f'train, keep 0 (every slot dropped): pred {rel(od["pred"], o1["pred"]):.1e} hist {rel(od["hist"], o1["hist"]):.1e} == R1')
rd.slot_keep = 0.5
seen = []
mem_fwd = rd.mem_net.forward
def mspy(h, keys, vals, mask=None):
    inj, a, g = mem_fwd(h, keys, vals, mask)
    seen.append((a.detach(), None if mask is None else mask.clone()))
    return inj, a, g
rd.mem_net.forward = mspy
torch.manual_seed(5)
rd.zero_grad(); rd.corr_aux = True
od = rd(x, N_OUT, perm=perm)
((od['pred'] - x[:, -N_OUT * P:]).square().mean() + rd.corr_loss).backward()
rd.mem_net.forward = mem_fwd
masks = [mk for _, mk in seen]
dropped_w = max((a * ~mk[:, None, :]).abs().max().item() for a, mk in seen)
sums_ok = all(torch.allclose(a.sum(-1), mk.any(1, keepdim=True).float().expand_as(a.sum(-1)), atol=1e-5) for a, mk in seen)
last = masks[-1]
frac = last.float().mean().item()
ok(all(mk is not None for mk in masks) and dropped_w == 0 and sums_ok,
   f'every retrieval masked; dropped slots get attention {dropped_w:.1e}; rows sum to 1 (0 if no slot survives)')
ok(0.2 < frac < 0.8 and (last != last[:1]).any().item(), f'masks are per sample (kept fraction of the full bank {frac:.2f}, samples differ)')
ok(all(torch.equal(masks[i][:, :masks[i].shape[1]], masks[-1][:, :masks[i].shape[1]]) for i in range(len(masks))),
   'one draw per slot: a dropped slot stays dropped for every later retrieval, history and rollout')
ok(all(torch.isfinite(p.grad).all() for p in rd.parameters() if p.grad is not None) and
   sum(p.grad.abs().sum().item() for p in rd.mem_net.parameters() if p.grad is not None) > 0, 'finite gradients, StateMem trained')

print('Q  low-rank Q/K (qk_dim): default == full-state; qk_dim r shrinks W_Q, W_K to D*N -> nh*r, same heads / value groups, trains')
_, rq0 = pair()
ok(tuple(rq0.mem_net.Wq.weight.shape) == (512, 512) and rq0.mem_net.dh == 64, f'default W_Q {tuple(rq0.mem_net.Wq.weight.shape)} dh {rq0.mem_net.dh}')
r1, rq = pair(qk_dim=8)
mn = rq.mem_net
nq = sum(p.numel() for n, p in mn.named_parameters() if n.startswith(('Wq', 'Wk')))
ok(tuple(mn.Wq.weight.shape) == (mn.nh * 8, 512) and tuple(mn.Wk.weight.shape) == (mn.nh * 8, 512) and mn.nh == 8 and nq == 2 * 512 * 64,
   f'W_Q {tuple(mn.Wq.weight.shape)} W_K {tuple(mn.Wk.weight.shape)}, {mn.nh} heads, W_Q+W_K {nq} params (full-state: {2 * 512 * 512})')
o1, oq = r1(x, N_OUT, perm=perm), rq(x, N_OUT, perm=perm)
ok(rel(oq['pred'], o1['pred']) < 1e-5, f'at init (C == 0) == R1 ({rel(oq["pred"], o1["pred"]):.1e})')
wake(rq)
rq.train(); rq.corr_aux = True; rq.zero_grad()
oq = rq(x, N_OUT, perm=perm)
((oq['pred'] - x[:, -N_OUT * P:]).square().mean() + rq.corr_loss).backward()
ok(rel(oq['pred'], o1['pred']) > 1e-3 and mn.Wq.weight.grad.abs().sum() > 0 and mn.Wk.weight.grad.abs().sum() > 0 and
   all(torch.isfinite(p.grad).all() for p in rq.parameters() if p.grad is not None), 'memory active changes the forecast; W_Q, W_K trained; finite grads')
rq.eval()

print('Z  small CorrNet code (z_dim): C keeps its (D, N) shape, Wc (D, z, N); zero at init (== R1); writer loss still CorrNet only')
r1, rz = pair(z_dim=8)
cn = rz.corr_net
ok(tuple(cn.Wc.shape) == (rz.D, 8, rz.N) and cn.E[-1].out_features == 8 and cn.G1.in_features == rz.N + 8,
   f'Wc {tuple(cn.Wc.shape)}, E out {cn.E[-1].out_features}, G1 in {cn.G1.in_features}; CorrNet {sum(p.numel() for p in cn.parameters())} params')
o1, oz = r1(x, N_OUT, perm=perm), rz(x, N_OUT, perm=perm)
ok(rel(oz['pred'], o1['pred']) < 1e-5, f'at init (Wc == 0 => C == 0) == R1 ({rel(oz["pred"], o1["pred"]):.1e})')
wake(rz)
Cs = []
fwd = cn.forward
cn.forward = lambda h, e: Cs.append(fwd(h, e)) or Cs[-1]
oz = rz(x, N_OUT, perm=perm)
cn.forward = fwd
ok(all(c.shape == (x.shape[0], rz.D, rz.N) for c in Cs) and rel(oz['pred'], o1['pred']) > 1e-3,
   f'C shape {tuple(Cs[0].shape)} at every slot; memory active changes the forecast')
rz.train(); rz.corr_aux = True; rz.zero_grad()
rz(x, N_OUT, perm=perm); rz.corr_loss.backward()
gp = grads_by_prefix(rz)
ok(gp['corr_net'] > 0 and all(v == 0 for k, v in gp.items() if k != 'corr_net'), f'L_corr ALONE reaches CorrNet only {gp}')
rz.eval()

print('U2 selection-free retrieval (uniform=True): no W_Q / W_K / tau; a = 1/available slots; == audit-uniform of the same weights; trains')
r1, ru = pair(uniform=True)
ok(not hasattr(ru.mem_net, 'Wq') and not hasattr(ru.mem_net, 'Wk') and not hasattr(ru.mem_net, 'log_tau'),
   f'no W_Q / W_K / tau; StateMem {sum(p.numel() for p in ru.mem_net.parameters())} params (gate only)')
wake(ru)
_, rn = pair()
rn.load_state_dict(ru.state_dict(), strict=False)                                    # same CorrNet / backbone / gate; rn keeps its own W_Q, W_K
rn.set_audit('uniform')
ok(rel(ru(x, N_OUT, perm=perm)['pred'], rn(x, N_OUT, perm=perm)['pred']) < 1e-6, 'uniform=True == audit "uniform" of the full model with the same other weights')
ru.train(); ru.slot_keep = 0.5; ru.corr_aux = True; ru.zero_grad()
seen = []
mem_fwd = ru.mem_net.forward
def uspy(h, keys, vals, mask=None):
    inj, a, g = mem_fwd(h, keys, vals, mask)
    seen.append((a.detach(), mask)); return inj, a, g
ru.mem_net.forward = uspy
torch.manual_seed(2)
ou = ru(x, N_OUT, perm=perm)
((ou['pred'] - x[:, -N_OUT * P:]).square().mean() + ru.corr_loss).backward()
ru.mem_net.forward = mem_fwd
ok(all(abs(a[b, :, ~mk[b]].sum().item()) == 0 for a, mk in seen for b in range(a.shape[0])) and
   all(torch.allclose(a.sum(-1), mk.any(1, keepdim=True).float().expand_as(a.sum(-1)), atol=1e-5) for a, mk in seen),
   'with slot dropout: dropped slots get 0, the rest share 1/(kept), rows sum to 1 (0 if none kept)')
ok(all(torch.isfinite(p.grad).all() for p in ru.parameters() if p.grad is not None) and ru.mem_net.Wg.weight.grad.abs().sum() > 0, 'finite grads, gate trained')

print('O  null slot (null=True): value-0 slot in every head softmax; == R1 at init; null mass > 0; abstains when b0 is large; trains')
r1, rn0 = pair(null=True)
mn = rn0.mem_net
ok(tuple(mn.k0.shape) == (mn.nh, mn.dh) and tuple(mn.b0.shape) == (mn.nh,), f'k0 {tuple(mn.k0.shape)}, b0 {tuple(mn.b0.shape)}')
o1, on = r1(x, N_OUT, perm=perm), rn0(x, N_OUT, perm=perm)
ok(rel(on['pred'], o1['pred']) < 1e-5, f'at init (C == 0) == R1 ({rel(on["pred"], o1["pred"]):.1e})')
wake(rn0)
seen = []
mem_fwd = mn.forward
def nspy(h, keys, vals, mask=None):
    inj, a, g = mem_fwd(h, keys, vals, mask)
    seen.append((a.detach(), keys.shape[1], mask)); return inj, a, g
mn.forward = nspy
on = rn0(x, N_OUT, perm=perm)
ok(all((a.sum(-1) < 1 - 1e-6).all() and (a.sum(-1) > 0).all() for a, M, _ in seen), 'real-slot weights sum to 1 - null mass, 0 < sum < 1')
with torch.no_grad():
    mn.b0.fill_(50.)
seen.clear()
ob = rn0(x, N_OUT, perm=perm)
ok(rel(ob['pred'], o1['pred']) < 1e-5 and all(a.abs().max() < 1e-12 for a, _, _ in seen), f'b0 large: all mass on null, forecast == R1 ({rel(ob["pred"], o1["pred"]):.1e})')
with torch.no_grad():
    mn.b0.zero_()
_, rf = pair()
rf.load_state_dict(rn0.state_dict(), strict=False)
with torch.no_grad():
    mn.b0.fill_(-50.)
ok(rel(rn0(x, N_OUT, perm=perm)['pred'], rf(x, N_OUT, perm=perm)['pred']) < 1e-5, 'b0 very negative (and k0 = 0): == model without null slot')
with torch.no_grad():
    mn.b0.zero_()
rn0.train(); rn0.slot_keep = 0.5; rn0.corr_aux = True; rn0.zero_grad(); seen.clear()
torch.manual_seed(3)
on = rn0(x, N_OUT, perm=perm)
((on['pred'] - x[:, -N_OUT * P:]).square().mean() + rn0.corr_loss).backward()
mn.forward = mem_fwd
ok(all(a[b, :, ~mk[b]].abs().sum().item() == 0 for a, _, mk in seen if mk is not None for b in range(a.shape[0])) and
   all(torch.isfinite(a).all() for a, _, _ in seen), 'with slot dropout: dropped slots get 0, all-dropped rows -> all mass on null, no NaN')
ok(mn.b0.grad is not None and mn.b0.grad.abs().sum() > 0 and mn.k0.grad.abs().sum() > 0 and
   all(torch.isfinite(p.grad).all() for p in rn0.parameters() if p.grad is not None), 'k0, b0 trained; finite grads')
rn0.eval()

print('L  R1_RO_LN=1: the writer loss still reaches CorrNet only (readout LayerNorm parameters stop-gradient too)')
r1, rm = pair(readout_ln=True)
wake(rm)
rm.train(); rm.corr_aux = True; rm.zero_grad()
rm(x, N_OUT, perm=perm); rm.corr_loss.backward()
gp = grads_by_prefix(rm)
ok(gp['corr_net'] > 0 and gp.get('ro_ln', 0) == 0 and all(v == 0 for k, v in gp.items() if k != 'corr_net'), f'grads {gp}')

print('D  diagnostics: diag forward == plain forward; recorded statistics finite; M = 1 entropy not recorded (N/A)')
r1, rm = pair()
wake(rm)
p_plain = rm(x, N_OUT, perm=perm)['pred']
rm.diag = True
p_diag = rm(x, N_OUT, perm=perm)['pred']
rm.diag = False
d = rm.pop_diag()
ok(rel(p_diag, p_plain) == 0, 'diag forward output identical')
need = ('C_fro', 'C_rms', 'C_over_hplus', 'writer_dL', 'writer_frac_improved', 'hist_gate_mean', 'hist_attn_entropy_norm', 'hist_top1_age',
        'hist_attn_eff_slots', 'roll_mean_age', 'roll_delta_over_hminus', 'hist_q_norm', 'hist_tau_mean')
ok(all(k in d and np.isfinite(d[k]) for k in need), 'keys present and finite: ' + ', '.join(f'{k}={d.get(k, float("nan")):.3g}' for k in need))
ok(rm.pop_diag() == {}, 'pop_diag clears the record')

print('Z  parameter snapshot: once per load event, all parameters + buffers, stats change after a load, nothing printed by forwards')
tmp = tempfile.mkdtemp(prefix='r1snap_')
lines = []
pf = lambda s, flush=True: lines.append(s)
r1, rm = pair()
st0 = log_model_parameter_stats(rm, 'init', tmp, meta=dict(test=1), print_fn=pf)
recs0 = [json.loads(l) for l in open(st0 + '.jsonl')]
names = {n for n, _ in rm.named_parameters()}
ok({r['name'] for r in recs0 if r['record'] == 'param'} == names, f'one row per named parameter ({len(names)})')
ok(any(r['record'] == 'param' and 'adjacency_matrix' in r['name'] and r['requires_grad'] is False for r in recs0),
   'VPT statistics (official nn.Parameter(requires_grad=False): adjacency / count / ending points) listed as frozen parameters')
ok(recs0[0]['record'] == 'header' and recs0[0]['extra_params_vs_R1'] == sum(p.numel() for n, p in rm.named_parameters()
                                                                         if n.startswith(('corr_net.', 'mem_net.'))), 'header: extra params vs R1')
ok(sum(n in '\n'.join(lines) for n in names) == len(names), 'console lists every parameter')
wake(rm)
sd = {k: v.clone() for k, v in rm.state_dict().items()}
rm.load_state_dict(sd)
st1 = log_model_parameter_stats(rm, 'checkpoint_loaded', tmp, meta=dict(checkpoint='in-memory'), print_fn=pf)
st2 = log_model_parameter_stats(rm, 'checkpoint_loaded', tmp, meta=dict(checkpoint='in-memory again'), print_fn=pf)
r_wc = lambda st: [json.loads(l) for l in open(st + '.jsonl') if '"corr_net.Wc"' in l][0]
ok(r_wc(st0)['l2'] == 0 and r_wc(st1)['l2'] > 0, f'Wc l2 init {r_wc(st0)["l2"]:.3g} -> after load {r_wc(st1)["l2"]:.3g}')
ok(st2 != st1 and os.path.exists(st2 + '.csv') and len(glob.glob(os.path.join(tmp, '*.jsonl'))) == 3, 'a repeated load prints again (3 snapshots)')
n_lines = len(lines)
rm(x, N_OUT, perm=perm); rm.train(); rm(x, N_OUT, perm=perm)
ok(len(lines) == n_lines and len(glob.glob(os.path.join(tmp, '*.jsonl'))) == 3, 'forwards print / write no snapshot')
bad = R1Mem(cfg()).to(dev)
with torch.no_grad():
    bad.readout.weight[0, 0] = float('nan')
st3 = log_model_parameter_stats(bad, 'init', tmp, print_fn=pf)
ok([json.loads(l) for l in open(st3 + '.jsonl') if '"readout.weight"' in l][0]['nonfinite_count'] == 1, 'non-finite entries counted, no crash')
shutil.rmtree(tmp)

print('C  checkpoint reload consistency: a fresh model loading the saved state_dict gives identical forecasts, the same ATSP eval order and')
print('   the same per-parameter snapshot; the reloaded model trains on identically (one Adam step, same batch, same RNG)')
tmp = tempfile.mkdtemp(prefix='r1ckpt_')
torch.manual_seed(11)
src = R1Mem(cfg()).to(dev)
wake(src, gate_bias=0.0)
src.train()
for _ in range(4):                                                                # populate the VPT transition statistics
    o = src(x, N_OUT)
    src.update_vpt_cost((o['pred'] - o['pred'].detach().mean()).square().detach())
ck = os.path.join(tmp, 'checkpoint.pth')
torch.save(src.state_dict(), ck)
torch.manual_seed(12345)
dst = R1Mem(cfg()).to(dev)                                                        # different init, then load
dst.load_state_dict(torch.load(ck))
src.eval(); dst.eval()
src.reset_ids_shuffle(); dst.reset_ids_shuffle()
np.random.seed(1); random.seed(1); solve_eval_order(src, 0)                       # different global RNG states before each solve
np.random.seed(2); np.random.rand(17); random.seed(5); random.random()
st_np, st_py = np.random.get_state()[1].copy(), random.getstate()
solve_eval_order(dst, 0)
ok(np.array_equal(np.random.get_state()[1], st_np) and random.getstate() == st_py, 'solve_eval_order restores the numpy and Python RNG states')
ps, pd = src(x, N_OUT)['pred'], dst(x, N_OUT)['pred']
ok(src.ids_shuffle == dst.ids_shuffle, f'same ATSP eval order {src.ids_shuffle} vs {dst.ids_shuffle} (deterministic solve)')
ok(rel(pd, ps) == 0, f'identical eval forecast after reload ({rel(pd, ps):.1e})')
rows = lambda st: [{k: v for k, v in json.loads(l).items()} for l in open(st + '.jsonl') if '"record": "param"' in l]
sa = log_model_parameter_stats(src, 'checkpoint_loaded', tmp, print_fn=pf)
sb = log_model_parameter_stats(dst, 'checkpoint_loaded', tmp, print_fn=pf)
ok(rows(sa) == rows(sb), 'per-parameter snapshot rows identical (saved vs reloaded)')
for mdl in (src, dst):
    mdl.train(); mdl.corr_aux = True
outs = []
for mdl in (src, dst):
    torch.manual_seed(99)
    opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
    opt.zero_grad()
    o = mdl(x, N_OUT, perm=perm)
    ((o['pred'] - x[:, -N_OUT * P:]).square().mean() + mdl.corr_loss).backward()
    opt.step()
    outs.append({n: p.detach().clone() for n, p in mdl.named_parameters()})
worst = max(rel(outs[1][n], outs[0][n]) for n in outs[0] if outs[0][n].norm() > 0)
ok(worst < 1e-6, f'one training step after reload matches the original (worst param rel diff {worst:.1e})')
sc = log_model_parameter_stats(dst, 'checkpoint_loaded', tmp, print_fn=pf)
ok(rows(sc) != rows(sb), 'snapshot after the update differs from the loaded one')
shutil.rmtree(tmp)

print('\n' + ('ALL PASS' if not FAILS else f'{len(FAILS)} FAIL(S): ' + '; '.join(FAILS)))
sys.exit(1 if FAILS else 0)
