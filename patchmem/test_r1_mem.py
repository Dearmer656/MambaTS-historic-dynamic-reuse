"""PAT-304 R1Mem gates: the memory design h+ = h- + dh_mem (C stored as the value only, never carried; C on the memory-corrected state;
retrieval over j < t; rollout retrieves without writing). usage: python patchmem/test_r1_mem.py"""
import sys
from argparse import Namespace
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.r1_model import R1
from patchmem.r1_mem import R1Mem

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


def pair(seed=0):
    """R1 and R1Mem with identical backbone / readout weights."""
    torch.manual_seed(seed)
    r1 = R1(cfg()).to(dev).eval()
    torch.manual_seed(seed + 100)
    rm = R1Mem(cfg()).to(dev).eval()
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
def spy(h, keys, vals):
    n_ret.setdefault(len(vals), 0); n_ret[len(vals)] += 1
    return orig(h, keys, vals)
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

print('\n' + ('ALL PASS' if not FAILS else f'{len(FAILS)} FAIL(S): ' + '; '.join(FAILS)))
sys.exit(1 if FAILS else 0)
