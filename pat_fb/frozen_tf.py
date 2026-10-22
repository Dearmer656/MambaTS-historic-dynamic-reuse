"""PAT-311 out-of-sample errors: the per-position error-feedback module of pat_fb/tf_timerxl.py (user design: error attached to the
predicting token j, PaTH path H_{j+1}..H_i, rollout memory = the lookback errors) on a FROZEN official Timer-XL checkpoint
(pat_slurm/repro.sbatch, the paper's recipe), trained on errors the backbone has NOT fitted.

Diagnosis that motivates it (pat_fb/diag_lead1.py): with in-sample errors (the backbone fits the training windows the errors come
from) the correction lowers the train MSE and raises the val MSE in every arm and seed.
--fit val (default): the module is fitted on the first half of the val windows and early-stopped on the rest after a gap of L + P
  steps (no shared time step between a fit window and a stop window; with a 70/30 cut the gap left ETTh1 only 68 stop windows);
  the backbone was never gradient-fitted on val (val only selected its epoch in phase 1), so every error is
  out-of-sample, as at test. Test untouched.
--fit train: the in-sample control (the module fitted on the train windows, early-stopped on val).
Loss: the official teacher-forced next-token MSE over the 7 positions of the corrected outputs (only the module is trained; position 0
is never corrected). Selection: the valid_last loss (the corrected last position) on the stop windows. Adam lr 1e-3, bs 32, <= 20
epochs, patience 3 (module only, constant lr). Baseline = the same checkpoint with the correction off (zero audit = the official run).
usage: python pat_fb/frozen_tf.py --data E1|W --setting <checkpoint dir> --arm errpath --rank 8 --keep 1 --fit val --seed 2021"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from pat_fb.fb_timerxl import CFG, HS, L, P, load_backbone, split_x, windows
from pat_fb.tf_timerxl import Module, evaluate, tf_pass


def split_ranges(n_val, n_train, fit, frac=0.5):
    """Window index ranges [lo, hi) of the fit and stop sets (in the split each window starts at its index)."""
    if fit == 'train':
        return ('train', 0, n_train), ('val', 0, n_val)
    cut = int(frac * n_val)
    assert cut + L + P + 32 <= n_val, f'val too short for a gap of L + P and >= 32 stop windows ({n_val})'
    return ('val', 0, cut), ('val', cut + L + P, n_val)


@torch.no_grad()
def stop_loss(bb, mod, xs, lo, hi, dev, bs=32):
    tot = 0.0
    for i0 in range(lo, hi, bs):
        x, y = windows(xs, list(range(i0, min(i0 + bs, hi))), P, dev)
        tot += (tf_pass(bb, mod, x)[0][:, -P:] - y).square().mean().item() * x.shape[0]
    return tot / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=['E1', 'W']); ap.add_argument('--setting', required=True)
    ap.add_argument('--arm', default='errpath'); ap.add_argument('--rank', type=int, default=8); ap.add_argument('--keep', type=float, default=1.0)
    ap.add_argument('--fit', default='val', choices=['val', 'train']); ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--bs', type=int, default=32); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    bb = load_backbone(a.data, a.setting, dev)                                              # frozen, eval (dropout off)
    mod = Module(a.arm, CFG[a.data]['d_model'], rank=a.rank, keep=a.keep).to(dev)
    xs = {f: split_x(a.data, f) for f in ('train', 'val', 'test')}
    nwin = {f: len(xs[f]) - L - P + 1 for f in xs}
    (ff, flo, fhi), (sf, slo, shi) = split_ranges(nwin['val'], nwin['train'], a.fit)
    tag = f'{a.arm}_r{a.rank}' + (f'_k{a.keep:g}' if a.keep < 1 else '') + f'_fit{a.fit}'
    npar = sum(p.numel() for p in mod.parameters())
    print(f'[frozen] {a.data} {tag} | module params {npar} | fit {ff}[{flo}:{fhi}] stop {sf}[{slo}:{shi}] | lr {a.lr}', flush=True)
    opt = torch.optim.Adam(mod.parameters(), lr=a.lr)
    best, bad, state = stop_loss(bb, mod, xs[sf], slo, shi, dev), 0, {k: v.clone() for k, v in mod.state_dict().items()}
    print(f'[frozen] epoch 0 (correction off) | stop valid_last {best:.5f}', flush=True)
    rng = np.random.default_rng(a.seed)
    for ep in range(a.epochs):
        mod.train(); t0 = time.time(); tl = []
        perm = flo + rng.permutation(fhi - flo)
        for idx in (perm[i0:i0 + a.bs] for i0 in range(0, len(perm), a.bs)):
            x, y = windows(xs[ff], idx.tolist(), P, dev)
            loss = (tf_pass(bb, mod, x)[0] - torch.cat([x[:, P:], y], 1)).square().mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
        mod.eval()
        v = stop_loss(bb, mod, xs[sf], slo, shi, dev)
        print(f'[frozen] epoch {ep + 1} {time.time() - t0:.0f}s | fit {np.mean(tl):.5f} | stop valid_last {v:.5f}', flush=True)
        if v < best:
            best, bad, state = v, 0, {k: t.clone() for k, t in mod.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    mod.load_state_dict(state); mod.eval()
    res = dict(data=a.data, setting=a.setting, arm=a.arm, rank=a.rank, keep=a.keep, fit=a.fit, seed=a.seed, params=npar, best_stop=best, test={})
    audits = {'t0r': ('zero',), 'o': ('zero', 'uniform'), 'errpath': ('zero', 'noH', 'uniform')}[a.arm]
    for H in HS:
        mse, mae = evaluate(bb, mod, xs['test'], H, dev)
        r = dict(mse=mse, mae=mae)
        for au in audits:
            mod.audit = au; r[au] = evaluate(bb, mod, xs['test'], H, dev)[0]; mod.audit = None
        res['test'][H] = r
        print(f'[frozen] test H{H} | mse {mse:.4f} mae {mae:.4f}' + ''.join(f' | {au} {r[au]:.4f}' for au in audits), flush=True)
    out = a.out or os.path.join(ROOT, 'pat_logs', 'frozen', f'{a.data}_{tag}_s{a.seed}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
