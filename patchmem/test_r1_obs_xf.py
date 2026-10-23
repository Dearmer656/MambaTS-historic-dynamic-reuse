"""PAT-307 checks for ErrPath-x (vpath, vsrc='x') and ErrPath-F (vpath h-driven + forget=True).
X1 at init R1Obs(x) == R1; X2 history values == explicit H(x_M) ... H(x_{j+2}) e_j products (H driven by the observed patch that comes
with each newer slot), rollout patch k values == H(fb_k) ... H(fb_1) applied to them (fb = the fed-back normalised forecast); keys = O's;
X3 lookback mask through forward: masked slots H = I; X4 noH == O bit for bit; X5 W_wv is C*P x C*P and gets a gradient.
F1 at init (alpha = 0) ErrPath-F == ErrPath with the same weights, bit for bit; F2 the D passed at rollout patch k ==
sum_{s > j} softplus(W_f e_s + b_f) (masked slots 0) + k softplus(c_roll); F3 the correction == the explicit softmax with the
alpha * D penalty; F4 audit noF == ErrPath bit for bit; F5 alpha, W_f, b_f, c_roll get gradients."""
import sys
from math import sqrt
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args, ETTH2
from patchmem.r1_model import R1
from patchmem.r1_obsmem import R1Obs

torch.manual_seed(0)
a = official_args(ETTH2 + ['--des', 'test'])
dev = 'cuda'
b, T, C, P = 3, 15, 7, 48; M = T - 1
x = torch.randn(b, 720, 7, device=dev)
r1 = R1(a).to(dev).eval()
perm = r1.scan_order(b, dev)


def capture(net):
    seen = []
    f0 = net.forward
    net.forward = lambda h, K, V, mask=None, D=None: (seen.append((h.clone(), K.clone(), V.clone(), None if D is None else D.clone())), f0(h, K, V, mask, D))[1]
    return seen, lambda: setattr(net, 'forward', f0)


def errors(m, tc, lens=None):
    bst = tc['bstates']; yh = m.to_orig(m.read(bst), perm)
    xn = m.revin(x, lens)[2]
    xo = xn.permute(0, 2, 1).reshape(b, C, T, P).permute(0, 2, 1, 3)
    return xo, xo[:, 1:] - yh[:, :-1]


def Hmat(net, z):
    z = z.flatten(1)
    w = F.normalize(net.Wwv(z), dim=-1); beta = 2 * torch.sigmoid(net.Wbv(z))
    return torch.eye(z.shape[1], device=dev) - beta[..., None] * w[:, :, None] * w[:, None, :]


# ---------------- ErrPath-x
ox = R1Obs(a, qk_dim=8, slot_keep=0.5, vpath=True, vsrc='x').to(dev).eval()
ox.load_state_dict(r1.state_dict(), strict=False)
with torch.no_grad():
    assert torch.equal(r1(x, 4, perm=perm)['pred'], ox(x, 4, perm=perm)['pred']); print('X1 init == R1')
    net = ox.obs_net
    net.W.normal_(0, 0.1); net.Wg.weight.normal_(0, 0.01); net.Wbv.bias.fill_(0.5)
    for lens in (None, torch.tensor([720, 480, 288], device=dev)):
        seen, undo = capture(net)
        tc = {}; out = ox(x, 3, perm=perm, trace=tc, lens=lens); undo()
        xo, E = errors(ox, tc, lens)
        means, stdev, _, t0, _ = ox.revin(x, lens)
        ok = torch.ones(b, M, dtype=torch.bool, device=dev) if lens is None else torch.arange(1, T, device=dev)[None, :] >= (t0 + 1)[:, None]
        eye = torch.eye(C * P, device=dev)
        Hs = [torch.where(ok[:, i, None, None], Hmat(net, xo[:, i + 1]), eye) for i in range(M)]
        ref = torch.empty_like(E.flatten(2))
        for j in range(M):
            v = E.flatten(2)[:, j]
            for s_ in range(j + 1, M):
                v = torch.einsum('bde,be->bd', Hs[s_], v)
            ref[:, j] = v
        outs = ((out['pred'] - means) / stdev).reshape(b, 3, P, C).transpose(-1, -2)    # normalised forecasts (b, k, C, P)
        K0 = net.key(tc['bstates'][:, 1:].flatten(0, 1)).unflatten(0, (b, M))
        for k in range(3):
            if k:
                Hf = Hmat(net, outs[:, k - 1])
                ref = torch.einsum('bde,bme->bmd', Hf, ref)
            err = (seen[k][2].flatten(2) - ref).abs().max().item(); kerr = (seen[k][1] - K0).abs().max().item()
            print(f'X2/X3 lens {None if lens is None else lens.tolist()} rollout patch {k}: values == explicit products {err:.2e} | keys {kerr:.1e}')
            assert err < 1e-4 and kerr == 0
    o = R1Obs(a, qk_dim=8, slot_keep=0.5).to(dev).eval()
    o.load_state_dict({k: v for k, v in ox.state_dict().items() if not k.startswith(('obs_net.Wwv', 'obs_net.Wbv'))})
    ox.set_audit('noH'); pn = ox(x, 4, perm=perm)['pred']; ox.set_audit(None)
    assert torch.equal(pn, o(x, 4, perm=perm)['pred']); print('X4 noH == O')
assert net.Wwv.weight.shape == (C * P, C * P)
ox.train(); (ox(x, 3)['pred'].square().mean()).backward()
assert net.Wwv.weight.grad is not None and net.Wwv.weight.grad.norm() > 0; print('X5 W_wv', tuple(net.Wwv.weight.shape), 'grad ok')

# ---------------- ErrPath-F
ep = R1Obs(a, qk_dim=8, slot_keep=0.5, vpath=True).to(dev).eval()
ep.load_state_dict(r1.state_dict(), strict=False)
with torch.no_grad():
    ep.obs_net.W.normal_(0, 0.1); ep.obs_net.Wg.weight.normal_(0, 0.01); ep.obs_net.Wbv.bias.fill_(0.5)
ef = R1Obs(a, qk_dim=8, slot_keep=0.5, vpath=True, forget=True).to(dev).eval()
ef.load_state_dict(ep.state_dict(), strict=False)
with torch.no_grad():
    assert torch.equal(ep(x, 4, perm=perm)['pred'], ef(x, 4, perm=perm)['pred']); print('F1 init (alpha 0) == ErrPath')
    nf = ef.obs_net
    nf.alpha.normal_(0, 0.5); nf.c_roll.fill_(0.3); nf.Wf.weight.normal_(0, 0.05)
    for lens in (None, torch.tensor([720, 480, 288], device=dev)):
        seen, undo = capture(nf)
        tc = {}; ef(x, 3, perm=perm, trace=tc, lens=lens); undo()
        _, E = errors(ef, tc, lens)
        t0 = ef.revin(x, lens)[3]
        ok = torch.ones(b, M, dtype=torch.bool, device=dev) if lens is None else torch.arange(1, T, device=dev)[None, :] >= (t0 + 1)[:, None]
        d = F.softplus(nf.Wf(E.flatten(2))) * ok[..., None]                             # (b, M, nh)
        for k in range(3):
            Dref = torch.stack([d[:, j + 1:].sum(1) for j in range(M)], 1) + k * F.softplus(nf.c_roll)
            err = (seen[k][3] - Dref).abs().max().item()
            h, K, V, D = seen[k]
            q = nf.Wq(h.flatten(1)).reshape(b, nf.nh, nf.dh)
            logit = torch.einsum('bhd,bmhd->bhm', q, K) / (sqrt(nf.dh) * nf.log_tau.exp())[None, :, None] - nf.alpha[None, :, None] * Dref.transpose(1, 2)
            if lens is not None:
                logit = logit.masked_fill(~ok[:, None, :], float('-inf'))
            aw = torch.softmax(logit, -1)
            g = torch.sigmoid(nf.Wg(h.flatten(1))).unsqueeze(-1)
            dref = g * (nf.W[None] * torch.einsum('bhm,bmcp->bhcp', aw, V)).sum(1)
            derr = (nf(h, K, V, ok if lens is not None else None, D)[0] - dref).abs().max().item()
            print(f'F2/F3 lens {None if lens is None else lens.tolist()} rollout patch {k}: D == explicit {err:.2e} | correction == explicit {derr:.2e}')
            assert err < 1e-5 and derr < 1e-5
    ef.train(); torch.manual_seed(1)                                                     # slot dropout (keep 0.5) in training
    seen, undo = capture(nf)
    masks = []
    f1 = nf.forward
    nf.forward = lambda h, K, V, mask=None, D=None: (masks.append(mask.clone()), f1(h, K, V, mask, D))[1]
    tc = {}; ef(x, 2, perm=perm, trace=tc); nf.forward = f1; undo()
    ef.eval()
    _, E = errors(ef, tc)
    mk = masks[0]
    assert (~mk).any() and mk.any()
    d = F.softplus(nf.Wf(E.flatten(2))) * mk[..., None]
    Dref = torch.stack([d[:, j + 1:].sum(1) for j in range(M)], 1)
    err = (seen[0][3] - Dref).abs().max().item()
    print(f'F2b training slot dropout ({(~mk).float().mean().item():.2f} dropped): D == explicit with dropped writes costing 0 {err:.2e}'); assert err < 1e-5
    ef.set_audit('noF'); pn = ef(x, 4, perm=perm)['pred']; ef.set_audit(None)
    ep2 = R1Obs(a, qk_dim=8, slot_keep=0.5, vpath=True).to(dev).eval()
    ep2.load_state_dict({k: v for k, v in ef.state_dict().items() if not k.startswith(('obs_net.Wf', 'obs_net.c_roll', 'obs_net.alpha'))})
    assert torch.equal(pn, ep2(x, 4, perm=perm)['pred']) and not torch.equal(pn, ef(x, 4, perm=perm)['pred']); print('F4 noF == ErrPath, forget changes forecast')
ef.train(); (ef(x, 3)['pred'].square().mean()).backward()
for n in ('alpha', 'Wf.weight', 'Wf.bias', 'c_roll'):
    g = dict(nf.named_parameters())[n].grad
    assert g is not None and g.norm() > 0, n
print('F5 grads ok | params x', sum(p.numel() for p in ox.obs_net.parameters()), 'F', sum(p.numel() for p in nf.parameters()))
print('ALL OK')
