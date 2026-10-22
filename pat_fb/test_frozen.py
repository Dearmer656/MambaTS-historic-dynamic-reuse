"""PAT-311 checks for pat_fb/frozen_tf.py. usage: python pat_fb/test_frozen.py <setting> <repro log>
1 the module with the correction off reproduces the official checkpoint's test MSE (H96, H336) from the repro log
2 fit / stop windows (fit=val) share no time step and lie inside val; fit=train uses the train split
3 one step: only module weights get gradients (backbone frozen, no grad), the backbone stays in eval mode"""
import re, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat_fb.fb_timerxl import CFG, L, P, load_backbone, split_x, windows
from pat_fb.tf_timerxl import Module, evaluate, tf_pass
from pat_fb.frozen_tf import split_ranges

setting, log = sys.argv[1], sys.argv[2]
D = 'E1' if 'ETTh1' in setting else 'W'
dev = torch.device('cuda')
bb = load_backbone(D, setting, dev)
mod = Module('errpath', CFG[D]['d_model'], keep=0.5).to(dev)
xs = {f: split_x(D, f) for f in ('train', 'val', 'test')}
txt = open(log).read(); parts = re.split(r'== H (\d+)', txt)
ref = {int(parts[i]): float(re.findall(r'mse:([0-9.]+)', parts[i + 1])[-1]) for i in range(1, len(parts), 2)}
mod.eval()
for H in (96, 336):
    m = evaluate(bb, mod, xs['test'], H, dev)[0]
    assert abs(m - ref[H]) < 1e-5 * ref[H], (H, m, ref[H])
print('1 ok: correction off (W2 = 0) == official checkpoint test MSE')
nv, nt = len(xs['val']) - L - P + 1, len(xs['train']) - L - P + 1
(ff, flo, fhi), (sf, slo, shi) = split_ranges(nv, nt, 'val')
assert ff == sf == 'val' and 0 <= flo < fhi <= slo < shi <= nv
assert (fhi - 1) + L + P - 1 < slo, 'fit and stop windows overlap'
assert split_ranges(nv, nt, 'train')[0] == ('train', 0, nt)
print(f'2 ok: fit val[{flo}:{fhi}] and stop val[{slo}:{shi}] disjoint in time; fit=train uses train')
mod.train()
torch.nn.init.normal_(mod.W2.weight, std=0.1)
x, y = windows(xs['val'], list(range(0, 8)), P, dev)
((tf_pass(bb, mod, x)[0] - torch.cat([x[:, P:], y], 1)).square().mean()).backward()
assert all(p.grad is None for p in bb.parameters()) and not bb.training
assert all(getattr(mod, n).weight.grad.abs().sum() > 0 for n in ('W2', 'W1', 'U', 'Wq', 'Wk', 'Ww', 'Wb'))
print('3 ok: only the module trains, backbone frozen in eval mode')
print('ALL OK')
