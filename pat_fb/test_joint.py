"""PAT-311 checks for pat_fb/joint_timerxl.py (GPU, random-init backbone). usage: python pat_fb/test_joint.py E1|W
1 ctrl rollout == the official test loop of the same backbone (eval), exactly
2 every arm == ctrl at init (W2 = 0), same backbone weights
3 errors: E[:, c, n-1] == (x_n - out_{n-1}) / sd of the same teacher-forced pass; the TF target == the official seq_y layout
4 errpath values at the origin == the explicit product H_{N-1} ... H_{j+2} U e_{j+1}; keys from token j+1; noH audit == arm 'o' with the same weights
5 the path continues: at step k the values are H(z_last(k)) applied to step k-1's values
6 one joint step: backbone AND module get gradients (module after W2 moves), errors / states detached in the module inputs
7 audits: zero == ctrl, uniform != trained"""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat_fb.fb_timerxl import CFG, L, P, split_x, windows
from pat_fb.joint_timerxl import Module, build_backbone, forward_states, roll_loss, run
from data_provider.data_loader import MultivariateDatasetBenchmark
import os

D = sys.argv[1]
dev = torch.device('cuda')
torch.manual_seed(0)
bb = build_backbone(D).to(dev).eval()
dm = CFG[D]['d_model']
xs = split_x(D, 'test')
x, y = windows(xs, list(range(0, 16)), 768, dev)
ctrl = Module('ctrl', dm).to(dev)

# 1 official loop (use_norm in place, unmodified forward)
with torch.no_grad():
    bb.use_norm = True
    bx, py = x.clone(), []
    for j in range(4):
        if py:
            bx = torch.cat([bx[:, P:], py[-1]], 1)
        py.append(bb(bx.clone(), None, None)[:, -P:])
    bb.use_norm = False
    assert torch.equal(torch.cat(py, 1), run(bb, ctrl, x, 4)[1]), 'ctrl != official loop'
print('1 ok: ctrl rollout == official test loop')

# 2 all arms == ctrl at init
with torch.no_grad():
    ref = run(bb, ctrl, x, 8)[1]
    for arm in ('t0r', 'o', 'errpath'):
        assert torch.equal(run(bb, Module(arm, dm).to(dev), x, 8)[1], ref), arm
print('2 ok: t0r / o / errpath == ctrl at init')

# 3 errors and the TF target
with torch.no_grad():
    out, sd, Z = forward_states(bb, x)
    mod = Module('errpath', dm).to(dev)
    torch.nn.init.normal_(mod.W2.weight, std=0.1)
    E = ((x[:, P:] - out[:, :-P]) / sd).reshape(16, L // P - 1, P, -1).permute(0, 3, 1, 2)
    for n in range(1, L // P):
        assert torch.allclose(E[:, :, n - 1], ((x[:, n * P:(n + 1) * P] - out[:, (n - 1) * P:n * P]) / sd).permute(0, 2, 1))
ds = MultivariateDatasetBenchmark(os.path.join(ROOT, CFG[D]['root_path']), flag='train', size=[L, P, P], data_path=CFG[D]['data_path'])
xt = split_x(D, 'train')
sx, sy = ds[5][0], ds[5][1]
wx, wy = windows(xt, [5], 768, 'cpu')
assert torch.allclose(torch.cat([wx[0, P:], wy[0, :P]], 0), torch.as_tensor(sy, dtype=torch.float32)), 'TF target != official seq_y'
print('3 ok: errors, TF target == official seq_y')

# 4 path product at the origin, keys, noH == o
with torch.no_grad():
    mem = mod.init_memory(E, Z)
    V0 = mod.U(E)
    N = L // P
    for j in range(N - 1):
        v = V0[:, :, j]
        for s_ in range(j + 2, N):
            v = mod.householder(Z[:, :, s_], v[:, :, None])[:, :, 0]
        assert torch.allclose(mem['V'][:, :, j], v, atol=1e-5), j
    assert torch.allclose(mem['K'], mod.Wk(Z[:, :, 1:]).unflatten(-1, (mod.nh, mod.dk)))
    o = Module('o', dm).to(dev); o.load_state_dict({k: v for k, v in mod.state_dict().items() if not k.startswith(('Ww', 'Wb'))})
    mod.audit = 'noH'
    assert torch.allclose(run(bb, mod, x, 8)[1], run(bb, o, x, 8)[1], atol=1e-6); mod.audit = None
print('4 ok: origin values == explicit Householder product, keys from token j+1, noH == o')

# 5 path continues in the rollout
with torch.no_grad():
    mem = mod.init_memory(E, Z)
    Vprev = mem['V'].clone()
    zk = torch.randn_like(Z[:, :, -1])
    mod.step(mem, 1, zk)
    assert torch.allclose(mem['V'], mod.householder(zk, Vprev))
print('5 ok: rollout step applies H(z_last) to the values')

# 6 joint gradients
bb.train(); mod.train()
xt_, yt_ = windows(xt, list(range(0, 8)), 768, dev)
out, pred = run(bb, mod, xt_, 8)
loss = (out - torch.cat([xt_[:, P:], yt_[:, :P]], 1)).square().mean() + roll_loss(pred, yt_)
loss.backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bb.parameters())
for n_ in ('W2', 'W1', 'U', 'Wq', 'Wk', 'Ww', 'Wb'):
    assert getattr(mod, n_).weight.grad is not None and getattr(mod, n_).weight.grad.abs().sum() > 0, n_
print('6 ok: backbone and every module weight get gradients')

# 7 audits
bb.eval(); mod.eval()
with torch.no_grad():
    mod.audit = 'zero'; assert torch.equal(run(bb, mod, x, 8)[1], run(bb, ctrl, x, 8)[1])
    mod.audit = 'uniform'; u = run(bb, mod, x, 8)[1]; mod.audit = None
    assert not torch.allclose(u, run(bb, mod, x, 8)[1])
print('7 ok: zero == ctrl, uniform differs')
print('ALL OK')
