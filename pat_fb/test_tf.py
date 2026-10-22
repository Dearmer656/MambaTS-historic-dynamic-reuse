"""PAT-311 checks for pat_fb/tf_timerxl.py (GPU, random-init backbone). usage: python pat_fb/test_tf.py E1|W
1 ctrl rollout == the official test loop, exactly; valid_last == the official vali (valid_last) loss
2 every arm == ctrl at init (teacher-forced outputs and rollout)
3 errpath position i == the explicit formula W2 ReLU(W1 [sum_{j<=i-1} softmax(q_i.k_j) H_i...H_{j+1} U e_{j+1}]) with k_j from token j
4 causality: the correction of position i does not change when the not-yet-observed errors e_{j+1}, j >= i, change
5 position 0 is not corrected; position i's correction lands on its own forecast (patch i+1)
6 rollout: the memory is the lookback errors only; at step k the values are H(z_last(k)) ... H(z_last(1)) of the origin values, keys fixed
7 t0r position i reads [e_{i-1}, e_i]; noH == o with the same weights
8 one training step: backbone and every module weight get gradients
9 error dropout: off at eval (keep < 1 module == keep 1 module, same weights); in training E[input] == the errors (Monte Carlo)"""
import sys
from math import sqrt
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat_fb.fb_timerxl import CFG, L, P, split_x, windows
from pat_fb.joint_timerxl import build_backbone, forward_states
from pat_fb.tf_timerxl import Module, rollout, tf_pass, valid_last

D_ = sys.argv[1]
dev = torch.device('cuda')
torch.manual_seed(0)
bb = build_backbone(D_).to(dev).eval()
dm = CFG[D_]['d_model']
xs = split_x(D_, 'test')
x, y = windows(xs, list(range(0, 16)), 768, dev)
N = L // P
ctrl = Module('ctrl', dm).to(dev)

# 1 official loop and official vali
with torch.no_grad():
    bb.use_norm = True
    bx, py = x.clone(), []
    for j in range(4):
        if py:
            bx = torch.cat([bx[:, P:], py[-1]], 1)
        py.append(bb(bx.clone(), None, None)[:, -P:])
    xv = split_x(D_, 'val')
    n = len(xv) - L - P + 1
    tot, cnt = 0.0, 0
    for i0 in range(0, n, 32):
        xb, yb = windows(xv, list(range(i0, min(i0 + 32, n))), P, dev)
        tot += (bb(xb.clone(), None, None)[:, -P:] - yb).square().mean().item() * xb.shape[0]; cnt += xb.shape[0]
    bb.use_norm = False
    assert torch.equal(torch.cat(py, 1), rollout(bb, ctrl, x, 4)), 'ctrl != official loop'
    assert abs(valid_last(bb, ctrl, xv, dev) - tot / cnt) < 1e-6
print('1 ok: ctrl rollout == official loop, valid_last == official vali')

# 2 init
with torch.no_grad():
    ref_tf, ref_ro = tf_pass(bb, ctrl, x)[0], rollout(bb, ctrl, x, 8)
    for arm in ('t0r', 'o', 'errpath'):
        m = Module(arm, dm).to(dev)
        assert torch.equal(tf_pass(bb, m, x)[0], ref_tf) and torch.equal(rollout(bb, m, x, 8), ref_ro), arm
print('2 ok: every arm == ctrl at init')

# 3 explicit formula
mod = Module('errpath', dm).to(dev)
with torch.no_grad():
    torch.nn.init.normal_(mod.W2.weight, std=0.1)
    out, sd, Z = forward_states(bb, x)
    E = ((x[:, P:] - out[:, :-P]) / sd).reshape(16, N - 1, P, -1).permute(0, 3, 1, 2)
    Dm, mem = mod.teacher_forced(E, Z)
    V0, K0 = mod.U(E), mod.Wk(Z[:, :, :-1]).unflatten(-1, (mod.nh, mod.dk))
    for i in range(1, N):
        vs = []
        for j in range(i):
            v = V0[:, :, j]
            for s_ in range(j + 1, i + 1):                                                   # H_{j+1} first, H_i last
                v = mod.householder(Z[:, :, s_], v[:, :, None])[:, :, 0]
            vs.append(v)
        Vt = torch.stack(vs, 2)
        q = mod.Wq(Z[:, :, i]).unflatten(-1, (mod.nh, mod.dk))
        a = torch.softmax(torch.einsum('bchd,bcmhd->bchm', q, K0[:, :, :i]) / (sqrt(mod.dk) * mod.log_tau.exp())[None, None, :, None], -1)
        d = mod.W2(torch.relu(mod.W1(torch.einsum('bchm,bcmr->bchr', a, Vt).flatten(2))))
        assert torch.allclose(Dm[:, :, i - 1], d, atol=1e-5), i
print('3 ok: errpath == explicit PaTH-convention formula (path H_{j+1}..H_i, key from token j)')

# 4 causality
with torch.no_grad():
    for i in range(1, N - 1):
        E2 = E.clone(); E2[:, :, i:] = torch.randn_like(E2[:, :, i:])                      # e_{j+1}, j >= i: not observed at i
        D2 = mod.teacher_forced(E2, Z)[0]
        assert torch.allclose(D2[:, :, :i], Dm[:, :, :i], atol=1e-6), i
print('4 ok: position i only reads errors observed by position i')

# 5 placement
with torch.no_grad():
    oc = tf_pass(bb, mod, x)[0]
    assert torch.equal(oc[:, :P], out[:, :P])
    for i in range(1, N):
        assert torch.allclose(oc[:, i * P:(i + 1) * P], out[:, i * P:(i + 1) * P] + Dm[:, :, i - 1].transpose(1, 2) * sd, atol=1e-6)
print('5 ok: position 0 uncorrected, position i corrected by its own delta')

# 6 rollout memory
with torch.no_grad():
    p = rollout(bb, mod, x, 3)
    assert torch.allclose(p[:, :P], oc[:, -P:])
    V, Kf = mem['V'].clone(), mem['K']
    xw = x
    prev = p[:, :P]
    for k in range(1, 3):
        xw = torch.cat([xw[:, P:], prev], 1)
        o_k, _, Z_k = forward_states(bb, xw)
        V = mod.householder(Z_k[:, :, -1], V)
        d = mod.read(Z_k[:, :, -1], V, Kf)
        step = o_k[:, -P:] + d.transpose(1, 2) * sd
        assert torch.allclose(p[:, k * P:(k + 1) * P], step, atol=1e-5), k
        prev = step
print('6 ok: rollout uses the lookback memory only, path continues through the new states, keys fixed')

# 7 t0r alignment, noH == o
with torch.no_grad():
    t = Module('t0r', dm).to(dev); torch.nn.init.normal_(t.W2.weight, std=0.1)
    Dt = t.teacher_forced(E, Z)[0]
    for i in range(1, N):
        prev_e = E[:, :, i - 2] if i >= 2 else torch.zeros_like(E[:, :, 0])
        assert torch.allclose(Dt[:, :, i - 1], t.W2(torch.relu(t.W1(torch.cat([prev_e, E[:, :, i - 1]], -1)))), atol=1e-6)
    o = Module('o', dm).to(dev); o.load_state_dict({k: v for k, v in mod.state_dict().items() if not k.startswith(('Ww', 'Wb'))})
    mod.audit = 'noH'
    assert torch.allclose(rollout(bb, mod, x, 8), rollout(bb, o, x, 8), atol=1e-6); mod.audit = None
print('7 ok: t0r reads [e_{i-1}, e_i]; noH == o')

# 8 gradients
bb.train(); mod.train()
xt, yt = windows(split_x(D_, 'train'), list(range(0, 8)), P, dev)
loss = (tf_pass(bb, mod, xt)[0] - torch.cat([xt[:, P:], yt], 1)).square().mean()
loss.backward()
assert any(p_.grad is not None and p_.grad.abs().sum() > 0 for p_ in bb.parameters())
for n_ in ('W2', 'W1', 'U', 'Wq', 'Wk', 'Ww', 'Wb'):
    assert getattr(mod, n_).weight.grad is not None and getattr(mod, n_).weight.grad.abs().sum() > 0, n_
print('8 ok: backbone and every module weight get gradients from L_tf')

# 9 error dropout
from pat_fb import tf_timerxl as T
mk = Module('t0r', dm, keep=0.5).to(dev); mk.load_state_dict(t.state_dict())
bb.eval()                                                                                   # test 8 left the backbone in train mode
with torch.no_grad():
    mk.eval(); t.eval()
    assert torch.equal(rollout(bb, mk, x, 8), rollout(bb, t, x, 8))
    seen = []
    orig = mk.teacher_forced
    mk.teacher_forced = lambda E_, Z_: (seen.append(E_), orig(E_, Z_))[1]
    mk.train()
    for _ in range(400):
        tf_pass(bb, mk, x[:4])
    mk.teacher_forced = orig
    Em = torch.stack(seen).mean(0)
    out4, sd4, Z4 = forward_states(bb, x[:4])
    E4 = ((x[:4, P:] - out4[:, :-P]) / sd4).reshape(4, N - 1, P, -1).permute(0, 3, 1, 2)
    rel = ((Em - E4).norm() / E4.norm()).item()
    zero_frac = torch.stack([(s_.abs().sum(-1) == 0).float().mean() for s_ in seen]).mean().item()
    assert rel < 0.1 and abs(zero_frac - 0.5) < 0.02, (rel, zero_frac)
print(f'9 ok: dropout off at eval; train E[input] == errors (rel err {rel:.3f}), dropped slot fraction {zero_frac:.3f}')
print('ALL OK')
