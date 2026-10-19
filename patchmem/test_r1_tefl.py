"""PAT-307 R1Tefl checks, both modes (native T0, faithful T1): (1) same-seed construction gives the R1 backbone weights exactly, and at
init (W2 = 0) R1Tefl == R1 element-wise; (2) eps against an independent reference: native = R1's own in-window next-patch errors over the
last Hc patches (R1Obs values V[:, -Hc:]), faithful = a separate R1 run on x[:, :L-Hc*P] rolled out Hc patches; (3) causality: the
faithful context forecast does not move when [t-Hc, t) is perturbed, the forecast of sample i does not depend on the other samples,
eps is detached; (4) feed=False: forecast - R1 = denormalised g(eps) exactly; (5) audits zero == R1, perm changes the forecast;
(6) gradients: W2 gets a gradient at init, W1 once W2 != 0."""
import sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args, ETTH2
from patchmem.r1_model import R1
from patchmem.r1_tefl import R1Tefl

a = official_args(ETTH2 + ['--des', 'test'])
dev = 'cuda'
torch.manual_seed(0); r1 = R1(a).to(dev).eval()
P, c, L, n_out, hc = r1.P, r1.c, 720, 4, 2
x = torch.randn(4, L, c, device=dev).cumsum(1) * 0.1 + torch.randn(4, 1, c, device=dev)
perm = torch.stack([torch.randperm(c) for _ in range(4)]).to(dev)
mx = lambda u, v: (u - v).abs().max().item()
with torch.no_grad():
    p1 = r1(x, n_out, perm=perm)['pred']
for mode in ('native', 'faithful'):
    torch.manual_seed(0); tf = R1Tefl(a, mode=mode, hc=hc).to(dev).eval()
    sd1, sdt = r1.state_dict(), tf.state_dict()
    assert all(torch.equal(sd1[k], sdt[k]) for k in sd1), 'backbone init differs from R1 under the same seed'
    assert set(sdt) - set(sd1) == {'W1.weight', 'W2.weight'}, set(sdt) - set(sd1)
    with torch.no_grad():
        o = tf(x, n_out, perm=perm, ret_eps=True)
        print(f'[{mode}] init forecast vs R1', mx(o['pred'], p1)); assert torch.equal(o['pred'], p1)
        eps = o['eps']
        assert not eps.requires_grad
        mu = x.mean(1, keepdim=True); sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        if mode == 'native':                                                            # in-window next-patch errors, last Hc patches
            h = r1(x, n_out, perm=perm)['hist']                                         # (b, T-1, P, C) predictions of patches 1..T-1
            tgt = x[:, P:].reshape(4, -1, P, c)
            ref = ((tgt - h) / sd.unsqueeze(1))[:, -hc:].permute(0, 3, 1, 2).reshape(4, c, hc * P)
        else:                                                                           # separate R1 run on the context only
            pc = r1(x[:, :L - hc * P], hc, perm=perm)['pred']
            ref = ((x[:, L - hc * P:] - pc) / sd).transpose(1, 2)
        print(f'[{mode}] eps vs reference', mx(eps, ref)); assert mx(eps, ref) < 1e-4
        x2 = x.clone(); x2[:, L - hc * P:] += torch.randn_like(x2[:, L - hc * P:])
        e2 = tf(x2, n_out, perm=perm, ret_eps=True)['eps']
        sd2 = torch.sqrt(x2.var(1, keepdim=True, unbiased=False) + 1e-5)
        if mode == 'faithful':                                                          # implied context forecast pc = x_tail - sd * eps
            pa = x[:, L - hc * P:] - sd * eps.transpose(1, 2); pb = x2[:, L - hc * P:] - sd2 * e2.transpose(1, 2)
            print(f'[{mode}] context forecast after perturbing [t-Hc, t)', mx(pa, pb)); assert mx(pa, pb) < 1e-4
        assert mx(e2, eps) > 1e-2
        tf.W2.weight.normal_(0, 0.05)
        pt = tf(x, n_out, perm=perm)['pred']
        assert mx(pt, p1) > 1e-4
        alone = torch.cat([tf(x[i:i + 1], n_out, perm=perm[i:i + 1])['pred'] for i in range(4)])
        print(f'[{mode}] batch independence', mx(alone, pt)); assert mx(alone, pt) < 1e-4
        tf.feed = False
        pf = tf(x, n_out, perm=perm)['pred']
        d = tf.W2(torch.relu(tf.W1(eps))).reshape(4, c, -1, P)[:, :, :n_out]            # (b, C, n_out, P)
        dd = (d.permute(0, 2, 3, 1) * sd.unsqueeze(1)).reshape(4, n_out * P, c)
        print(f'[{mode}] feed=False: forecast - R1 - g(eps)', mx(pf - p1, dd)); assert mx(pf - p1, dd) < 1e-4
        tf.feed = True
        tf.set_audit('zero'); pz = tf(x, n_out, perm=perm)['pred']; tf.set_audit(None)
        print(f'[{mode}] zero audit vs R1', mx(pz, p1)); assert mx(pz, p1) < 1e-5
        don = tf(torch.randn_like(x).cumsum(1) * 0.1, n_out, perm=perm, ret_eps=True)['eps']   # errors of unrelated windows
        tf.set_audit('perm'); pp = tf(x, n_out, perm=perm, eps_override=don)['pred']; tf.set_audit(None)
        assert mx(pp, pt) > 1e-4
    tf.W2.weight.data.zero_(); tf.train()
    y = torch.randn(4, n_out * P, c, device=dev)
    (tf(x, n_out, perm=perm)['pred'] - y).square().mean().backward()
    print(f'[{mode}] grad at init: W2 {tf.W2.weight.grad.norm().item():.3e} W1 {tf.W1.weight.grad.norm().item():.3e}')
    assert tf.W2.weight.grad.norm() > 0
    tf.zero_grad(); tf.W2.weight.data.normal_(0, 0.05)
    (tf(x, n_out, perm=perm)['pred'] - y).square().mean().backward()
    print(f'[{mode}] grad W2 != 0: W1 {tf.W1.weight.grad.norm().item():.3e}'); assert tf.W1.weight.grad.norm() > 0

# hc=0 (T0 = O without retrieval): eps = every O slot value, right-aligned and zero-padded to Tmax-1 patches
T, tmax = L // P, 30
torch.manual_seed(0); t0m = R1Tefl(a, mode='native', hc=0, tmax=tmax).to(dev).eval()
assert set(t0m.state_dict()) - set(r1.state_dict()) == {'W1.weight', 'W2.weight'} and t0m.din == (tmax - 1) * P
with torch.no_grad():
    o = t0m(x, n_out, perm=perm, ret_eps=True)
    assert torch.equal(o['pred'], p1)
    h = r1(x, n_out, perm=perm)['hist']
    mu = x.mean(1, keepdim=True); sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
    V = (x[:, P:].reshape(4, -1, P, c) - h) / sd.unsqueeze(1)                       # (b, T-1, P, C) = R1Obs values
    ref = torch.cat([V.new_zeros(4, tmax - T, P, c), V], 1).permute(0, 3, 1, 2).reshape(4, c, -1)
    print('[hc=0] eps vs all O slot values (right-aligned, zero-padded)', mx(o['eps'], ref)); assert mx(o['eps'], ref) < 1e-4
    # padded batch (lens) == unpadded windows, eps included
    lens = torch.tensor([L, L - 4 * P, L - 7 * P, L - 2 * P])
    ob = t0m(x, n_out, perm=perm, ret_eps=True, lens=lens)
    t0m.W2.weight.normal_(0, 0.05)
    ob = t0m(x, n_out, perm=perm, ret_eps=True, lens=lens)
    for i in range(4):
        oi = t0m(x[i:i + 1, -int(lens[i]):], n_out, perm=perm[i:i + 1], ret_eps=True)
        assert mx(ob['eps'][i:i + 1], oi['eps']) < 1e-4 and mx(ob['pred'][i:i + 1], oi['pred']) < 1e-3, i
    print('[hc=0] padded batch == unpadded windows (eps, forecast)')
    # extrapolation beyond Hmax = 7 patches
    pl = t0m(x, 10, perm=perm)['pred']
    t0m.feed = False
    for ext in ('last', 'zero'):
        t0m.extrap = ext
        pe = t0m(x, 10, perm=perm, ret_eps=True)
        d = t0m.W2(torch.relu(t0m.W1(pe['eps']))).reshape(4, c, 7, P)
        d = torch.cat([d, (d[:, :, -1:] if ext == 'last' else torch.zeros_like(d[:, :, -1:])).expand(4, c, 3, P)], 2)
        dd = (d.permute(0, 2, 3, 1) * sd.unsqueeze(1)).reshape(4, 10 * P, c)
        e = mx(pe['pred'] - r1(x, 10, perm=perm)['pred'], dd); print(f'[hc=0] extrap {ext}: forecast - R1 - delta', e); assert e < 1e-4
    t0m.feed = True; t0m.extrap = 'last'
    # perm audit needs donor errors; with donors = own errors it equals the normal forecast
    t0m.set_audit('perm')
    try:
        t0m(x, n_out, perm=perm); raise SystemExit('perm audit without eps_override must fail')
    except AssertionError:
        pass
    own = t0m(x, n_out, perm=perm, ret_eps=True, eps_override=o['eps'])['pred']
    t0m.set_audit(None)
    assert mx(own, t0m(x, n_out, perm=perm)['pred']) < 1e-6
    x3 = torch.randn_like(x).cumsum(1) * 0.1
    don = t0m(x3, n_out, perm=perm, ret_eps=True)['eps']
    assert mx(t0m(x, n_out, perm=perm, eps_override=don)['pred'], t0m(x, n_out, perm=perm)['pred']) > 1e-4
    print('[hc=0] perm audit: requires donors, own donors == normal, distant donors change the forecast')
    t0m.set_audit('recent'); pr = t0m(x, n_out, perm=perm)['pred']; t0m.set_audit(None)
    e_r = o['eps'].clone(); e_r[..., :-2 * P] = 0
    assert mx(pr, t0m(x, n_out, perm=perm, eps_override=e_r)['pred']) < 1e-6
    print('[hc=0] recent audit == eps with all but the last 2 patches zeroed')
print('OK')
