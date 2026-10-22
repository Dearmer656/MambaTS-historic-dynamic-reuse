"""PAT-311 diagnostic: does the trained module help the in-distribution (lead-1, last-position) forecast on train / val / test?
For each checkpoint: MSE of the last position's next-patch forecast with vs without the correction (zero audit), and the correction's
size ||delta*sd|| / ||residual||. usage: python pat_fb/diag_lead1.py E1"""
import sys, json
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat_fb.fb_timerxl import CFG, L, P, split_x, windows
from pat_fb.joint_timerxl import build_backbone
from pat_fb.tf_timerxl import Module, tf_pass

D = sys.argv[1]
dev = torch.device('cuda')
xs = {f: split_x(D, f) for f in ('train', 'val', 'test')}
for tag, arm, keep in [('t0r_r8', 't0r', 1.0), ('t0r_r8_k0.5', 't0r', 0.5), ('o_r8', 'o', 1.0), ('errpath_r8', 'errpath', 1.0), ('errpath_r8_k0.5', 'errpath', 0.5)]:
    for s in (2021, 2022, 2023):
        ck = torch.load(ROOT / 'checkpoints' / 'pat311_tf' / f'{D}_{tag}_s{s}.pth', map_location=dev)
        bb = build_backbone(D).to(dev); bb.load_state_dict(ck['bb']); bb.eval()
        m = Module(arm, CFG[D]['d_model'], keep=keep).to(dev); m.load_state_dict(ck['mod']); m.eval()
        row = []
        with torch.no_grad():
            for f in ('train', 'val', 'test'):
                n = len(xs[f]) - L - P + 1
                se1 = se0 = sd2 = r2 = 0.0
                for i0 in range(0, n, 128):
                    x, y = windows(xs[f], list(range(i0, min(i0 + 128, n))), P, dev)
                    p1 = tf_pass(bb, m, x)[0][:, -P:]
                    m.audit = 'zero'; p0 = tf_pass(bb, m, x)[0][:, -P:]; m.audit = None
                    se1 += (p1 - y).square().sum().item(); se0 += (p0 - y).square().sum().item()
                    sd2 += (p1 - p0).square().sum().item(); r2 += (p0 - y).square().sum().item()
                row.append(f'{f} {se1 / se0 - 1:+.2%} (|d|/|res| {(sd2 / r2) ** .5:.3f})')
        print(f'[diag] {D} {tag} s{s} | lead-1 MSE change from the correction: ' + ' | '.join(row), flush=True)
