"""PAT-315 stage 1: does the observed y-path between a demo and the query carry
error-predictive information beyond lag/phase, error similarity and endpoints?

Frozen standalone iTransformer (PAT-314 checkpoints).  For query origin t and
channel c, demos are matured past forecasts at s_j = t - age_j (age_j >= H):
key k_j = error of the forecast made at s_j - H (window ending at s_j - 1),
value e_j = error of the forecast made at s_j.  Query key q = error of the
forecast made at t - H.  Correction (attention, value untransformed):

    e_hat_t = s * sum_j softmax_j(score_{t,j}) e_j

  A      score = 0                                   (uniform mean, learned scale)
  B      b(lag, time marks at t-1 and s_j-1)
  Bp     B + phi(q)^T phi(k_j) / sqrt(d)
  End    Bp + g(y window at t-1, y window at s_j-1)
  C      B + phi(q)^T P_j phi(k_j) / sqrt(d),   P_j = A_{t-1} ... A_{s_j}
  Crev   same factors in reverse order: A_{s_j} ... A_{t-1} = P_j^T
  Cerr   C with A_r generated from realized lead-0 errors instead of y

A_r = I - beta_r u_r u_r^T, |u_r| = 1, beta_r = 2 sigmoid(.) in (0, 2), from a
causal window of y (or errors) ending at r <= t-1.  Parameters are shared over
channels and trained on the train split; val is used only for early stopping;
test is scored once.  Everything used for query t is observable before t.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

L = 96
H = int(os.environ.get('ITR_H', '192'))  # forecast horizon; set ITR_H for H != 192
D_EMB, WIN, HID = 16, 8, 32
VARIANTS = ('A', 'B', 'Bp', 'End', 'C', 'Crev', 'Cerr')
CELLS = {'T': dict(history=2880, period=168, stride=24)}


def demo_ages(kind, history, period, stride):
    last = history - H - L
    if kind == 'phase':
        first = ((H + period - 1) // period) * period
        return list(range(first, last + 1, period))
    return list(range(H, last + 1, stride))


# ----------------------------------------------------------------------------- data
@torch.inference_mode()
def load_split(exp, split, history, batch_size):
    """Backbone errors E[row = origin - L] (H x C), observed series X, time marks."""
    from data_provider.data_factory import data_provider
    args = exp.args
    old = args.seq_len
    args.seq_len = history
    try:
        ds, _ = data_provider(args, split)
    finally:
        args.seq_len = old
    X = torch.as_tensor(np.asarray(ds.data_x), dtype=torch.float32)
    marks = torch.as_tensor(np.asarray(ds.data_stamp), dtype=torch.float32)
    origins = torch.arange(L, len(X) - H + 1)
    E = torch.empty((len(origins), H, X.shape[1]), dtype=torch.float32)
    offsets, leads = torch.arange(-L, 0), torch.arange(H)
    for start in range(0, len(origins), batch_size):
        o = origins[start:start + batch_size]
        pred = exp.model(X[o[:, None] + offsets].to(exp.device), marks[o[:, None] + offsets].to(exp.device), None, None)
        E[start:start + len(o)] = X[o[:, None] + leads] - pred.cpu()
    official = torch.arange(history, history + len(ds))
    assert official[-1] + H == len(X)
    return dict(E=E, X=X, M=marks, origins=official)


def gather(data, t, c, ages, need_path):
    """Batch tensors for query origins t (B,) and channels c (B,); all strictly causal for t."""
    E, X, M = data['E'], data['X'], data['M']
    ages = torch.as_tensor(ages, dtype=torch.long)
    s = t[:, None] - ages[None]                                     # B n
    cc = c[:, None]
    out = dict(q=E[t - H - L, :, c], target=E[t - L, :, c],          # B H
               K=E[s - H - L, :, cc], V=E[s - L, :, cc],              # B n H
               mt=M[t - 1], ms=M[s - 1], ages=ages)
    w = torch.arange(WIN - 1, -1, -1)
    # endpoint y windows (ending at t-1 and at s_j-1)
    out['yt'] = X[(t - 1)[:, None] - w, c[:, None]]                  # B WIN
    out['ys'] = X[(s - 1)[..., None] - w, cc[..., None]]              # B n WIN
    if need_path:
        kmax = int(ages.max())
        r = t[:, None] - 1 - torch.arange(kmax)                      # B K   (k = 0 -> r = t-1)
        out['yr'] = X[r[..., None] - w, cc[..., None]]                # B K WIN
        # realized lead-0 error at time r (forecast made at r, known once y_r is seen, r <= t-1)
        out['er'] = E[r[..., None] - w - L, 0, cc[..., None]]         # B K WIN
    return out


# ----------------------------------------------------------------------------- model
def lag_features(ages, mt, ms):
    a = ages.to(mt.dtype)[None, :].expand(ms.shape[0], -1)          # B n
    f = [torch.log1p(a / H)]
    for p in (24.0, 168.0):
        f += [torch.sin(2 * math.pi * a / p), torch.cos(2 * math.pi * a / p)]
    f = torch.stack(f, -1)                                           # B n 5
    return torch.cat([f, mt[:, None].expand(-1, a.shape[1], -1), ms], -1)


def window_features(w):
    return torch.cat([w, torch.diff(w, dim=-1, prepend=w[..., :1])], -1)   # ... 2*WIN


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, HID), nn.GELU(), nn.Linear(HID, o))


def reflect(v, u, beta):
    """(I - beta u u^T) v for unit u; v, u: ... d, beta: ..."""
    return v - beta[..., None] * u * (u * v).sum(-1, keepdim=True)


class ErrorPath(nn.Module):
    def __init__(self, variant, n_marks):
        super().__init__()
        self.variant = variant
        self.scale = nn.Parameter(torch.tensor(0.3))
        if variant != 'A':
            self.lag = mlp(5 + 2 * n_marks, 1)
        if variant not in ('A', 'B'):
            self.phi = nn.Linear(H, D_EMB, bias=False)
        if variant == 'End':
            self.end = mlp(4 * WIN, 1)
        if variant in ('C', 'Crev', 'Cerr'):
            self.trans = mlp(2 * WIN, D_EMB + 1)

    def transitions(self, b):
        src = b['er'] if self.variant == 'Cerr' else b['yr']
        z = self.trans(window_features(src))                         # B K d+1
        return F.normalize(z[..., :-1], dim=-1), 2 * torch.sigmoid(z[..., -1])

    def path_scores(self, fq, fk, ages, u, beta):
        """C: fq^T P_j fk_j with P_j = A_{t-1}...A_{s_j} (age_j factors, k = 0..age_j-1).
        Crev: fq^T P_j^T fk_j.  Returns scores (B n) and |(P_j^T - I) fq| / |fq|."""
        n = len(ages)
        stop = (ages - 1).tolist()                                   # last k included for demo j
        if self.variant != 'Crev':
            # P_j^T fq = A_{s_j} ... A_{t-1} fq: apply k = 0 (r = t-1) first, record after k = age_j - 1
            w, rec = fq, [None] * n
            order = sorted(range(n), key=lambda j: stop[j])
            pos = 0
            for k in range(stop[order[-1]] + 1):
                w = reflect(w, u[:, k], beta[:, k])
                while pos < n and stop[order[pos]] == k:
                    rec[order[pos]] = w
                    pos += 1
            wq = torch.stack(rec, 1)                                 # B n d
            dev = (wq - fq[:, None]).norm(dim=-1) / fq.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return (wq * fk).sum(-1), dev
        # P_j^T fk_j = A_{s_j} ... A_{t-1} fk_j: same sweep on the keys, frozen after k = age_j - 1
        v = fk
        stop_t = torch.as_tensor(stop, device=fk.device)
        for k in range(max(stop) + 1):
            new = reflect(v, u[:, k, None], beta[:, k, None])
            v = torch.where((k <= stop_t)[None, :, None], new, v)
        dev = (v - fk).norm(dim=-1) / fk.norm(dim=-1).clamp_min(1e-8)
        return (fq[:, None] * v).sum(-1), dev

    def forward(self, b):
        V = b['V']
        diag = {}
        if self.variant == 'A':
            score = V.new_zeros(V.shape[:2])
        else:
            score = self.lag(lag_features(b['ages'].to(V.device), b['mt'], b['ms']))[..., 0]
        if self.variant in ('Bp', 'End'):
            score = score + (self.phi(b['q'])[:, None] * self.phi(b['K'])).sum(-1) / math.sqrt(D_EMB)
        if self.variant == 'End':
            yt = window_features(b['yt'])[:, None].expand(-1, V.shape[1], -1)
            score = score + self.end(torch.cat([yt, window_features(b['ys'])], -1))[..., 0]
        if self.variant in ('C', 'Crev', 'Cerr'):
            u, beta = self.transitions(b)
            ps, dev = self.path_scores(self.phi(b['q']), self.phi(b['K']), b['ages'], u, beta)
            score = score + ps / math.sqrt(D_EMB)
            diag = dict(beta=beta.detach(), dev=dev.detach())
        att = torch.softmax(score, -1)
        return self.scale * (att[..., None] * V).sum(1), diag


# ----------------------------------------------------------------------------- train / eval
def to_dev(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def mse_eval(model, data, ages, device, bs, max_cases=None, seed=0):
    """Sum of squared errors (backbone, corrected) over every (origin, channel) case."""
    C = data['X'].shape[1]
    t_all = data['origins'].repeat_interleave(C)
    c_all = torch.arange(C).repeat(len(data['origins']))
    if max_cases is not None and len(t_all) > max_cases:
        idx = torch.randperm(len(t_all), generator=torch.Generator().manual_seed(seed))[:max_cases]
        t_all, c_all = t_all[idx], c_all[idx]
    need = model.variant in ('C', 'Crev', 'Cerr')
    se_base = se = 0.0
    lead_base = torch.zeros(H, dtype=torch.float64)
    lead = torch.zeros(H, dtype=torch.float64)
    betas, devs = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(t_all), bs):
            b = to_dev(gather(data, t_all[i:i + bs], c_all[i:i + bs], ages, need), device)
            corr, diag = model(b)
            r0 = b['target'].double()
            r1 = (b['target'] - corr).double()
            se_base += float(r0.square().sum())
            se += float(r1.square().sum())
            lead_base += r0.square().sum(0).cpu()
            lead += r1.square().sum(0).cpu()
            if diag and len(betas) < 50:
                betas.append(diag['beta'].flatten()[::97].cpu())
                devs.append(diag['dev'].cpu())
    n = len(t_all) * H
    out = dict(cases=len(t_all), mse_base=se_base / n, mse=se / n,
               lead_gain_pct=(100 * (lead_base - lead) / lead_base).tolist())
    if betas:
        bq = torch.cat(betas).float()
        dq = torch.cat(devs).float()
        qs = torch.tensor([0.1, 0.5, 0.9])
        out['beta_q10_50_90'] = torch.quantile(bq, qs).tolist()
        out['dev_q10_50_90'] = torch.quantile(dq.flatten(), qs).tolist()
        out['dev_by_demo_median'] = dq.median(0).values.tolist()
    return out


def run(code, seed, kind, variant, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    torch.manual_seed(seed)
    exp, checkpoint, setting = load_base(code, seed)
    device = exp.device
    cell = CELLS[code]
    ages = demo_ages(kind, cell['history'], cell['period'], cell['stride'])
    splits = {s: load_split(exp, s, cell['history'], args.forecast_bs) for s in ('train', 'val', 'test')}
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    test_base = float(splits['test']['E'][splits['test']['origins'] - L].double().square().mean())
    if abs(test_base - official) > 1e-4:
        raise AssertionError(f'base cached={test_base} official={official}')
    del exp
    torch.cuda.empty_cache()
    model = ErrorPath(variant, splits['train']['M'].shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr = splits['train']
    C = tr['X'].shape[1]
    gen = torch.Generator().manual_seed(seed + 7)
    need = variant in ('C', 'Crev', 'Cerr')
    best, best_state, bad, log = float('inf'), None, 0, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for it in range(args.iters):
            ti = tr['origins'][torch.randint(len(tr['origins']), (args.bs,), generator=gen)]
            ci = torch.randint(C, (args.bs,), generator=gen)
            b = to_dev(gather(tr, ti, ci, ages, need), device)
            corr, _ = model(b)
            loss = (b['target'] - corr).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        val = mse_eval(model, splits['val'], ages, device, args.eval_bs, args.val_cases, seed)
        log.append(dict(epoch=epoch, train_loss=tot / args.iters, val_mse=val['mse'], val_base=val['mse_base'],
                        scale=float(model.scale), sec=time.time() - t0))
        print(f'[epoch {epoch}] train={tot / args.iters:.5f} val={val["mse"]:.6f} (base {val["mse_base"]:.6f}) '
              f's={float(model.scale):.3f} {time.time() - t0:.0f}s', flush=True)
        if val['mse'] < best - 1e-6:
            best, bad = val['mse'], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    test = mse_eval(model, splits['test'], ages, device, args.eval_bs, args.test_cases, seed)
    test['gain_pct'] = 100 * (test['mse_base'] - test['mse']) / test['mse_base']
    print(f'[test] {variant} {kind} mse={test["mse"]:.6f} base={test["mse_base"]:.6f} gain={test["gain_pct"]:+.2f}%'
          + (f' beta={test.get("beta_q10_50_90")} dev={test.get("dev_q10_50_90")}' if 'beta_q10_50_90' in test else ''),
          flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, demos=kind, variant=variant, ages=ages, d=D_EMB, win=WIN,
               checkpoint=str(checkpoint), train_cases_per_epoch=args.iters * args.bs, args=vars(args),
               log=log, best_val=best, test=test)
    output = Path(args.out_dir) / f'{code}_H{H}_{kind}_{variant}_s{seed}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T', choices=CELLS)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--demos', default='dense', choices=('dense', 'phase'))
    p.add_argument('--variant', required=True, choices=VARIANTS)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--iters', type=int, default=400)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--eval_bs', type=int, default=2048)
    p.add_argument('--val_cases', type=int, default=40000)
    p.add_argument('--test_cases', type=int, default=None)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a.demos, a.variant, a)


if __name__ == '__main__':
    main()
