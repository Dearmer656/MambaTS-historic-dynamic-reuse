"""PAT-311 in-context error feedback on a FROZEN Timer-XL (causal next-token forecaster), stage 2 of a two-phase recipe:
phase 1 = the reproduced Timer-XL checkpoint (pat_slurm/repro.sbatch, untouched), phase 2 = only the feedback module is trained here
(as patchmem/plug_a0.py on frozen MambaTS A0). The baseline of every arm is therefore the SAME backbone, seed by seed (paired).

Rolling forecast = the official test loop (exp/exp_forecast.py): window x (L = N*P), step 0 outputs bb(x)[:, -P:], step k >= 1 drops
the oldest patch and appends the previous forecast. Here the fed-back patch is the CORRECTED forecast (feed=True, as R1Tefl / R1Obs).

In-context errors (one forward pass, teacher-forced, as R1's history): Timer-XL token n (per variable) predicts patch n+1, so
  e_n = (x_n - bb(x)_{token n-1}) / sd,   n = 1..N-1  (patch index), sd = the window's RevIN std (Timer-XL's use_norm statistics)
e_n is the backbone's error, detached. Caveat (Timer-XL's own design): use_norm uses the WHOLE lookback's mean/std, so the teacher-
forced prediction of patch n saw the window statistics (which include patch n); the errors are in-window quantities, never future.
Arms (only the module is trained; y'_k = base_k + sd * delta_k, delta in the normalised scale, per variable, shared across variables):
Model selection: the module is early-stopped on the val rollout MSE over Hmax*P = 768 steps (one checkpoint for every reported H, as
the official one-for-all Timer-XL); the step-0 teacher-forced pass that gives the errors is also the base step-0 forecast (no extra pass).
  base  delta = 0                                                                                   (reproduces the Timer-XL test MSE)
  t0r   delta = W2 ReLU(W1 eps), eps = the last hc lookback errors (hc*P), W1: hc*P -> r (default init, no bias), W2: r -> Hmax*P
        (zero: base at init), rollout step k gets the k-th P-block                       (= patchmem/r1_tefl.py mode 'native', low rank r)
        hc = 0: all N-1 lookback errors (T0).
Audits: 'zero' (delta = 0, must equal base).
usage: python pat_fb/fb_timerxl.py --data E1|W --setting <checkpoint dir> --arm t0r --rank 8 --hc 2 --seed 2021"""
import argparse, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models import timer_xl
from data_provider.data_loader import MultivariateDatasetBenchmark

P, L = 96, 672
CFG = {'E1': dict(root_path='./dataset/ETT-small/', data_path='ETTh1.csv', e_layers=1, d_model=1024, d_ff=2048),
       'W': dict(root_path='./dataset/weather/', data_path='weather.csv', e_layers=4, d_model=512, d_ff=2048),
       'Wc': dict(root_path='./dataset/weather/', data_path='weather.csv', e_layers=4, d_model=512, d_ff=2048)}
HS = (96, 192, 336, 720)
HMAX = -(-max(HS) // P)                                                                     # 8 rollout steps cover H = 720


def load_backbone(D, setting, dev):
    c = CFG[D]
    cfg = SimpleNamespace(input_token_len=P, output_token_len=P, d_model=c['d_model'], d_ff=c['d_ff'], n_heads=8, e_layers=c['e_layers'],
                          dropout=0.1, activation='relu', output_attention=False, covariate=False, flash_attention=False, use_norm=True)
    m = timer_xl.Model(cfg)
    m.load_state_dict(torch.load(os.path.join(ROOT, 'checkpoints', setting, 'checkpoint.pth'), map_location='cpu'))
    m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    m.use_norm = False                                                                     # normalisation done by bb_forward (out of place)
    return m


def bb_forward(bb, x):
    """Timer-XL forecast with its use_norm done out of place (the official `x /= stdev` is in place and breaks backprop through a
    fed-back forecast); the same operations in the same order: returns (out (b, L, C), stdev (b, 1, C))."""
    means = x.mean(1, keepdim=True).detach()
    xc = x - means
    stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return bb.forecast(xc / stdev, None, None) * stdev + means, stdev


def split_x(D, flag):
    """The split's scaled series exactly as the official loader builds it (val/test include the L-step lookback prefix)."""
    c = CFG[D]
    ds = MultivariateDatasetBenchmark(os.path.join(ROOT, c['root_path']), flag=flag, size=[L, P, P], data_path=c['data_path'])
    return torch.as_tensor(ds.data_x, dtype=torch.float32)


def windows(xs, idx, Hf, dev):
    """Windows starting at idx: x (b, L, C), y (b, Hf, C)."""
    i = torch.as_tensor(idx)[:, None]
    x = xs[i + torch.arange(L)[None]]
    y = xs[i + L + torch.arange(Hf)[None]]
    return x.to(dev), y.to(dev)


class Feedback(nn.Module):
    def __init__(self, arm, rank=8, hc=2, n_tok=L // P, hmax=HMAX):
        super().__init__()
        assert arm in ('base', 't0r'), arm
        assert 0 <= hc <= n_tok - 1, f'hc must be in 0..{n_tok - 1} ({hc})'
        self.arm, self.hc, self.hmax, self.audit = arm, hc, hmax, None
        if arm == 't0r':
            self.din = (hc if hc else n_tok - 1) * P
            self.W1 = nn.Linear(self.din, rank, bias=False)
            self.W2 = nn.Linear(rank, hmax * P, bias=False)
            nn.init.zeros_(self.W2.weight)

    def delta(self, E):
        """E (b, C, N-1, P) lookback errors (normalised, detached) -> delta (b, C, Hmax, P) or None."""
        if self.arm == 'base' or self.audit == 'zero':
            return None
        eps = (E[:, :, -self.hc:] if self.hc else E).flatten(2)
        b, c = eps.shape[:2]
        return self.W2(torch.relu(self.W1(eps))).reshape(b, c, self.hmax, P)


def lookback_errors(bb, x):
    """Teacher-forced pass: out0 (b, L, C) (token n predicts patch n+1), E (b, C, N-1, P) = (x_n - pred_n) / sd, sd (b, 1, C)."""
    b, _, c = x.shape
    out0, sd = bb_forward(bb, x)                                                            # sd = Timer-XL's use_norm stdev
    sd = sd.detach()
    E = ((x[:, P:] - out0[:, :-P]) / sd).reshape(b, L // P - 1, P, c).permute(0, 3, 1, 2)
    return out0, E.detach(), sd


def rollout(bb, fb, x, n_out):
    """Rolling forecast of n_out patches (b, n_out*P, C) in the scaled data units, corrected forecasts fed back."""
    out0, E, sd = lookback_errors(bb, x)
    delta = fb.delta(E)
    preds, xw, base = [], x, out0[:, -P:]
    for k in range(n_out):
        if k:
            xw = torch.cat([xw[:, P:], preds[-1]], 1)
            base = bb_forward(bb, xw)[0][:, -P:]
        preds.append(base if delta is None else base + delta[:, :, min(k, fb.hmax - 1)].transpose(1, 2) * sd)
    return torch.cat(preds, 1)


@torch.no_grad()
def evaluate(bb, fb, xs, H, dev, bs=64):
    """Official test metric for horizon H: all windows with H future steps, MSE / MAE over (windows, H, C)."""
    n = len(xs) - L - H + 1
    se, ae = 0.0, 0.0
    for i0 in range(0, n, bs):
        x, y = windows(xs, list(range(i0, min(i0 + bs, n))), H, dev)
        p = rollout(bb, fb, x, -(-H // P))[:, :H]
        se += (p - y).square().sum().item(); ae += (p - y).abs().sum().item()
    nel = n * H * xs.shape[1]
    return se / nel, ae / nel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, choices=list(CFG)); ap.add_argument('--setting', required=True)
    ap.add_argument('--arm', default='t0r'); ap.add_argument('--rank', type=int, default=8); ap.add_argument('--hc', type=int, default=2)
    ap.add_argument('--seed', type=int, default=2021); ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--bs', type=int, default=32); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    bb = load_backbone(a.data, a.setting, dev)
    xs = {f: split_x(a.data, f) for f in ('train', 'val', 'test')}
    fb = Feedback(a.arm, rank=a.rank, hc=a.hc).to(dev)
    npar = sum(p.numel() for p in fb.parameters())
    Hf = HMAX * P
    ntr, nva = len(xs['train']) - L - Hf + 1, len(xs['val']) - L - Hf + 1
    tag = f'{a.arm}' + (f'_r{a.rank}_hc{a.hc}' if a.arm != 'base' else '')
    print(f'[fb] {a.data} arm {tag} | params {npar} | train windows {ntr} val windows {nva} (L {L}, fit horizon {Hf})', flush=True)
    res = dict(data=a.data, setting=a.setting, arm=a.arm, rank=a.rank, hc=a.hc, seed=a.seed, params=npar)
    if npar:
        opt = torch.optim.Adam(fb.parameters(), lr=a.lr)
        best, bad, state = float('inf'), 0, None
        rng = np.random.default_rng(a.seed)
        for ep in range(a.epochs):
            fb.train(); t0 = time.time(); tl = []
            for idx in np.array_split(rng.permutation(ntr), max(ntr // a.bs, 1)):
                x, y = windows(xs['train'], idx.tolist(), Hf, dev)
                loss = (rollout(bb, fb, x, HMAX) - y).square().mean()
                opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
            fb.eval()
            v = evaluate(bb, fb, xs['val'], Hf, dev)[0]
            print(f'[fb] epoch {ep + 1} {time.time() - t0:.0f}s | train {np.mean(tl):.5f} | val mse (H{Hf}) {v:.5f}', flush=True)
            if v < best:
                best, bad, state = v, 0, {k: t.clone() for k, t in fb.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        assert state is not None, 'no epoch was run'
        fb.load_state_dict(state)
        res['best_val'] = best
    fb.eval()
    res['test'] = {}
    for H in HS:
        mse, mae = evaluate(bb, fb, xs['test'], H, dev)
        r = dict(mse=mse, mae=mae)
        if npar:
            fb.audit = 'zero'; r['zero'] = evaluate(bb, fb, xs['test'], H, dev)[0]; fb.audit = None
        res['test'][H] = r
        print(f'[fb] test H{H} | mse {mse:.4f} mae {mae:.4f}' + (f' | zero audit {r["zero"]:.4f}' if npar else ''), flush=True)
    out = a.out or os.path.join(ROOT, 'pat_logs', 'fb', f'{a.data}_{tag}_s{a.seed}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
