"""PAT-311 checks for pat_fb/fb_timerxl.py on a reproduced Timer-XL checkpoint (GPU). usage: python pat_fb/test_fb.py <setting> <repro log>
1 rollout(base) == the official exp/exp_forecast.py test loop on a batch (H = 336, 4 steps), exactly
2 evaluate(base) reproduces the repro log's test MSE for H = 96 and H = 336 (all windows)
3 the lookback errors: E[:, c, n-1] == (x_n - out0_{token n-1}) / sd, recomputed from the unmodified Timer-XL forward
4 t0r == base at init (W2 = 0); one Adam step gives W2 a gradient and changes the forecast; audit 'zero' == base
5 feed-back: with W2 != 0 the step-1 backbone input is the CORRECTED step-0 forecast
6 window counts == the official loader's len for test (H 96..720); H = 720 uses 8 steps truncated to 720; hc = 0 path; backbone gets no grad"""
import re, sys
from pathlib import Path
import torch
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat_fb.fb_timerxl import (L, P, Feedback, bb_forward, evaluate, load_backbone, lookback_errors, rollout, split_x, windows)

setting, log = sys.argv[1], sys.argv[2]
D = 'E1' if 'ETTh1' in setting else ('Wc' if 'cos' in setting else 'W')
dev = torch.device('cuda')
torch.manual_seed(0)
bb = load_backbone(D, setting, dev)
official = lambda z: (setattr(bb, 'use_norm', True), bb(z.clone(), None, None), setattr(bb, 'use_norm', False))[1]   # unmodified Timer-XL forward
xs = split_x(D, 'test')
base = Feedback('base').to(dev)

# 1 official loop
x, y = windows(xs, list(range(0, 64)), 336, dev)
with torch.no_grad():
    bx, pred_y = x.clone(), []
    for j in range(4):
        if pred_y:
            bx = torch.cat([bx[:, P:, :], pred_y[-1]], dim=1)
        pred_y.append(official(bx)[:, -P:, :])
    off = torch.cat(pred_y, 1)[:, :336]
    ours = rollout(bb, base, x, 4)[:, :336]
assert torch.equal(off, ours), (off - ours).abs().max()
print('1 ok: rollout(base) == official test loop')

# 2 repro log
txt = open(log).read()
parts = re.split(r'== H (\d+)', txt)
ref = {int(parts[i]): float(re.findall(r'mse:([0-9.]+)', parts[i + 1])[-1]) for i in range(1, len(parts), 2)}
for H in (96, 336):
    m = evaluate(bb, base, xs, H, dev)[0]
    assert abs(m - ref[H]) < 1e-5 * ref[H], (H, m, ref[H])
    print(f'2 ok: H{H} base {m:.6f} == repro log {ref[H]:.6f}')

# 3 errors
with torch.no_grad():
    out0, E, sd = lookback_errors(bb, x)
    full = official(x)
    for n in range(1, L // P):
        e = (x[:, n * P:(n + 1) * P] - full[:, (n - 1) * P:n * P]) / sd                    # (b, P, C)
        assert torch.allclose(E[:, :, n - 1], e.permute(0, 2, 1), atol=1e-6)
print('3 ok: lookback errors (token n-1 predicts patch n of the official forward)')

# 4 init / gradient / zero audit
fb = Feedback('t0r', rank=8, hc=2).to(dev)
with torch.no_grad():
    assert torch.equal(rollout(bb, fb, x, 4), rollout(bb, base, x, 4))
xt, yt = windows(split_x(D, 'train'), list(range(0, 32)), 768, dev)
opt = torch.optim.Adam(fb.parameters(), lr=1e-3)
loss = (rollout(bb, fb, xt, 8) - yt).square().mean()
opt.zero_grad(); loss.backward()
assert fb.W2.weight.grad.abs().sum() > 0 and fb.W1.weight.grad.abs().sum() == 0        # W2 = 0 blocks W1's gradient at step 0
opt.step()
with torch.no_grad():
    assert not torch.equal(rollout(bb, fb, x, 4), rollout(bb, base, x, 4))
    fb.audit = 'zero'
    assert torch.equal(rollout(bb, fb, x, 4), rollout(bb, base, x, 4)); fb.audit = None
print('4 ok: t0r == base at init, trains, zero audit == base')

# 5 feed-back of the corrected forecast
with torch.no_grad():
    nn_ = torch.nn
    nn_.init.normal_(fb.W2.weight, std=0.05)
    p = rollout(bb, fb, x, 2)
    out0, E, sd = lookback_errors(bb, x)
    d = fb.delta(E)
    c0 = out0[:, -P:] + d[:, :, 0].transpose(1, 2) * sd
    assert torch.allclose(p[:, :P], c0, atol=1e-6)
    b1 = official(torch.cat([x[:, P:], c0], 1))[:, -P:]
    assert torch.allclose(p[:, P:], b1 + d[:, :, 1].transpose(1, 2) * sd, atol=1e-5)
print('5 ok: corrected forecast fed back, step k gets block k')

# 6 window counts / H = 720 / hc = 0 / frozen backbone
from data_provider.data_loader import MultivariateDatasetBenchmark
from pat_fb.fb_timerxl import CFG, ROOT as R
import os
for H in (96, 192, 336, 720):
    ds = MultivariateDatasetBenchmark(os.path.join(R, CFG[D]['root_path']), flag='test', size=[L, P, H], data_path=CFG[D]['data_path'])
    assert len(ds) == len(xs) - L - H + 1, (H, len(ds))
    i = len(ds) - 1
    sx, sy = ds[i][0], ds[i][1]
    wx, wy = windows(xs, [i], H, 'cpu')
    assert torch.allclose(wx[0], torch.as_tensor(sx, dtype=torch.float32)) and torch.allclose(wy[0], torch.as_tensor(sy, dtype=torch.float32)[-H:])
with torch.no_grad():
    x7, y7 = windows(xs, list(range(0, 8)), 720, dev)
    p8 = rollout(bb, fb, x7, 8)
    assert p8.shape[1] == 768 and torch.equal(p8[:, :720], rollout(bb, fb, x7, 8)[:, :720])
f0 = Feedback('t0r', rank=8, hc=0).to(dev)
torch.nn.init.normal_(f0.W2.weight, std=0.01)
assert f0.W1.in_features == (L // P - 1) * P
loss = (rollout(bb, f0, xt, 8) - yt).square().mean(); loss.backward()
assert f0.W1.weight.grad.abs().sum() > 0 and all(p.grad is None for p in bb.parameters())
print('6 ok: test window counts / targets == official loader, H720 = 8 steps truncated, hc=0 path, backbone gets no grad')
print('ALL OK')
