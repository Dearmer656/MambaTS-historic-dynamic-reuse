"""PAT-307 multi-step residual checks (patchmem/r1_msres.py), eval mode:
(1) boundary_layer_states: last layer == R1.history boundary states, every layer's final state == history's per-layer states;
(2) multistep_residuals == a per-origin reference (history on the prefix tokens of origin j, then K free-running steps), and lead 1 ==
    O's one-step values; leads beyond the lookback are masked;
(3) causality: perturbing patch p changes no prediction of an origin j < p (only residual entries whose TARGET is p);
(4) R1ObsMS / R1TeflMS: init == R1 exactly, zero audit == R1, the forecast loss reaches the memory / g parameters;
    T0-MS eps == the reference leads of origin T-1-K; lens: origins before t0 are unavailable."""
import sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args, ETTH2
from patchmem.r1_model import R1
from patchmem.r1_msres import boundary_layer_states, multistep_residuals, R1ObsMS, R1TeflMS

torch.manual_seed(0)
a = official_args(ETTH2 + ['--des', 'test'])
dev = 'cuda'
r1 = R1(a).to(dev).eval()
b, L, K = 3, 720, 7
P, c = r1.P, r1.c
T = L // P
x = torch.randn(b, L, c, device=dev)
perm = r1.scan_order(b, dev)

def prep(m, x):
    means, stdev, xn, t0, tm = m.revin(x)
    xo = xn.permute(0, 2, 1).reshape(b, c, T, P).permute(0, 2, 1, 3)
    tok = m.embed(m.to_scan(xo, perm)).reshape(b, T * c, -1)
    return xo, tok, t0, tm

with torch.no_grad():
    xo, tok, t0, tm = prep(r1, x)
    bst, sts = r1.history(tok)
    caps = boundary_layer_states(r1, tok)
    e1 = (caps[-1] - bst).abs().max().item(); e2 = max((cp[:, -1] - s).abs().max().item() for cp, s in zip(caps, sts))
    print(f'(1) last layer == history bstates {e1:.2e} | per-layer final == history states {e2:.2e}'); assert e1 < 1e-4 and e2 < 1e-4
    E, avail = multistep_residuals(r1, tok, xo, perm, tm, t0, K)
    yh = r1.to_orig(r1.read(bst), perm)
    e3 = (E[:, :, 0] - (xo[:, 1:] - yh[:, :-1])).abs().max().item()
    print(f'(2a) lead 1 == one-step values {e3:.2e}'); assert e3 < 1e-4
    worst = 0.0
    for j in range(T - 1):                                                               # per-origin reference rollout
        _, st = r1.history(tok[:, :(j + 1) * c])
        base = r1.to_orig(r1.read(st[-1]).unsqueeze(1), perm).squeeze(1); preds = [base]
        for k in range(1, K):
            st = r1.step_time(r1.embed(r1.to_scan(preds[-1].unsqueeze(1), perm)).squeeze(1), st)
            preds.append(r1.to_orig(r1.read(st[-1]).unsqueeze(1), perm).squeeze(1))
        for k in range(K):
            tgt = j + 1 + k
            assert bool(avail[:, j, k].all()) == (tgt <= T - 1)
            if tgt <= T - 1:
                worst = max(worst, (E[:, j, k] - (xo[:, tgt] - preds[k])).abs().max().item())
            else:
                assert E[:, j, k].abs().max() == 0
    print(f'(2b) batched multi-step residuals == per-origin reference {worst:.2e}; masked leads beyond the lookback'); assert worst < 1e-3
    p = 8                                                                               # (3) causality, in the normalised space
    xo2 = xo.clone(); xo2[:, p] += 1.0
    tok2 = r1.embed(r1.to_scan(xo2, perm)).reshape(b, T * c, -1)
    E2, _ = multistep_residuals(r1, tok2, xo2, perm, tm, t0, K)
    j = torch.arange(T - 1)[:, None]; tgt = j + 1 + torch.arange(K)[None, :]
    early = (j < p) & (tgt != p)
    d_early = (E2 - E)[:, early.to(dev)].abs().max().item()
    d_tgt = (E2 - E)[:, ((j < p) & (tgt == p)).to(dev)].abs().max().item()
    print(f'(3) origins j < {p}: non-target entries unchanged {d_early:.2e}, target-{p} entries change {d_tgt:.3f}'); assert d_early < 1e-5 and d_tgt > 0.5

for name, m in (('O-MS', R1ObsMS(a, qk_dim=8, slot_keep=0.5, kmax=K)), ('T0-MS', R1TeflMS(a, hmax=K))):
    m = m.to(dev).eval(); m.load_state_dict(r1.state_dict(), strict=False)
    with torch.no_grad():
        p1, p2 = r1(x, 4, perm=perm)['pred'], m(x, 4, perm=perm)['pred']
        print(f'(4) {name} init == R1', (p1 - p2).abs().max().item()); assert torch.equal(p1, p2)
        if name == 'O-MS':
            m.obs_net.W.normal_(0, 0.1); m.obs_net.Wg.weight.normal_(0, 0.01)
        else:
            m.W2.weight.normal_(0, 0.01)
            eps = m(x, 4, perm=perm, ret_eps=True)['eps']                                # (b, C, K*P)
            ref = E[:, T - 1 - K].permute(0, 2, 1, 3).reshape(b, c, K * P)
            print(f'    T0-MS eps == leads of origin T-1-K {(eps - ref).abs().max().item():.2e}'); assert (eps - ref).abs().max() < 1e-4
        pc = m(x, 4, perm=perm)['pred']; assert (pc - p1).abs().max() > 1e-5
        m.set_audit('zero'); pz = m(x, 4, perm=perm)['pred']; m.set_audit(None)
        print(f'    zero audit == R1 {(pz - p1).abs().max().item():.2e}; correction changes forecast {(pc - p1).abs().max().item():.3f}')
        assert (pz - p1).abs().max() < 1e-5
    m.train()
    out = m(x, 3); y = torch.randn_like(out['pred']); (out['pred'] - y).square().mean().backward()
    names = ('W', 'Wq.weight', 'Wk.weight', 'Wg.weight') if name == 'O-MS' else ('W1.weight', 'W2.weight')
    src = dict(m.obs_net.named_parameters()) if name == 'O-MS' else dict(m.named_parameters())
    for n in names:
        g = src[n].grad; print(f'    grad {n}', None if g is None else f'{g.norm().item():.2e}'); assert g is not None and g.norm() > 0

with torch.no_grad():                                                                    # lens: origins before t0 unavailable
    lens = torch.tensor([720, 480, 384], device=dev)
    means, stdev, xn, t0, tm = r1.revin(x, lens)
    xo3 = xn.permute(0, 2, 1).reshape(b, c, T, P).permute(0, 2, 1, 3)
    tok3 = r1.embed(r1.to_scan(xo3, perm)).reshape(b, T * c, -1)
    _, av3 = multistep_residuals(r1, tok3, xo3, perm, tm, t0, K)
    for i in range(b):
        assert not bool(av3[i, :int(t0[i])].any()) and bool(av3[i, int(t0[i]), 0])
    print('(5) lens: origins before t0 masked, first valid origin available; t0 =', t0.tolist())
    om = R1ObsMS(a, qk_dim=8, kmax=K).to(dev).eval(); om.load_state_dict(r1.state_dict(), strict=False)
    om.obs_net.W.normal_(0, 0.1)
    assert torch.isfinite(om(x, 7, perm=perm, lens=lens)['pred']).all()
print('OK')
