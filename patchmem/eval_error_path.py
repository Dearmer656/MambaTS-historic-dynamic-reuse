"""PAT-315 stage 1: does the observed y-path between a demo and the query carry
error-predictive information beyond lag/phase, error similarity and endpoints?

Time axis (one channel; E_o = backbone error vector of the forecast issued at origin o,
covering y_o .. y_{o+H-1}; that forecast reads y_{o-L} .. y_{o-1}):
  query at origin t      : q   = E_{t-H}   (window y_{t-H} .. y_{t-1}, last observed time t-1)
                           target = E_t    (y_t .. y_{t+H-1}; never an input)
  demo j, s_j = t - age_j: k_j = E_{s_j-H} (last observed time s_j-1)
                           e_j = E_{s_j}   (value; matured since s_j + H <= t, age_j >= H)
  P_j transports the key state from its last observed time s_j-1 to the query's t-1:
      P_j = A_{t-1} A_{t-2} ... A_{s_j}        (age_j factors; A_r reads y_{r-7} .. y_r, r <= t-1)
  Every input for query t only uses y_r with r <= t-1 and E entries E_o[h] with o + h <= t-1.

Correction (attention; the value is not transformed, so this stage tests
path-conditioned *retrieval/weighting*, not error transport):

    e_hat_t = s * sum_j softmax_j(score_{t,j}) e_j,     qn = RMSNorm(phi(q)), kn_j = RMSNorm(phi(k_j))

  A      score = 0                                        (uniform mean, learned s)
  B      b(lag, time marks at t-1 and s_j-1)
  Bp     B + qn^T kn_j / sqrt(d)
  End    Bp + g(y window at t-1, y window at s_j-1)       (endpoints only)
  C      B + qn^T P_j kn_j / sqrt(d)
  Cmean  B + qn^T diag(2 sigmoid(mean_{r in path_j} psi(y window at r))) kn_j / sqrt(d)
         (same per-step inputs and capacity as C, but a commutative summary of the path)
  Crev   C with the factor order reversed: A_{s_j} ... A_{t-1} = P_j^T   (diagnostic only)
  Cerr   C with A_r generated from realized lead-0 errors instead of y

A_r = I - beta_r u_r u_r^T, u_r = l2-normalised (as PaTH's w), beta_r = 2 (exact Householder
reflection, norm preserving; --beta two, main) or 2 sigmoid(.) in (0, 2) (PaTH; --beta learn).
q/k use RMSNorm as PaTH's qk-norm.  Parameters are shared over channels and trained on the
train split (in-sample backbone residuals; a known limitation), val is used only for early
stopping, test is scored once.  For C, a factor-order permutation of the trained model
(per demo, same factor set and endpoints) is evaluated on a test subset as a diagnostic.
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
VARIANTS = ('A', 'B', 'Bp', 'End', 'C', 'Cmean', 'Crev', 'Cerr')
PATH_VARIANTS = ('C', 'Crev', 'Cerr')
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


def residual_stats(data):
    r = data['E'][data['origins'] - L].double()
    return dict(mean=float(r.mean()), std=float(r.std()), mse=float(r.square().mean()))


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
    out['yt'] = X[(t - 1)[:, None] - w, c[:, None]]                  # B WIN   (ends at t-1)
    out['ys'] = X[(s - 1)[..., None] - w, cc[..., None]]              # B n WIN (ends at s_j-1)
    if need_path:
        kmax = int(ages.max())
        r = t[:, None] - 1 - torch.arange(kmax)                      # B K   (k = 0 <-> r = t-1)
        out['yr'] = X[r[..., None] - w, cc[..., None]]                # B K WIN
        # realized lead-0 error at time r: forecast issued at r, y_r observed (r <= t-1)
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


def factor_order(ages, mode, gen=None):
    """Per-demo factor index sequence: step i applies factor k = idx[j, i] (valid for i < age_j).
    'P'   : P_j v   = A_{t-1} ... A_{s_j} v  -> apply k = age_j-1 first, k = 0 last
    'PT'  : P_j^T v = A_{s_j} ... A_{t-1} v  -> apply k = 0 first
    'perm': a random order of the same factors (fixed generator), per demo."""
    K = int(max(ages))
    idx = torch.zeros(len(ages), K, dtype=torch.long)
    for j, a in enumerate(ages):
        a = int(a)
        if mode == 'P':
            seq = torch.arange(a - 1, -1, -1)
        elif mode == 'PT':
            seq = torch.arange(a)
        else:
            seq = torch.randperm(a, generator=gen)
        idx[j, :a] = seq
    return idx


def sweep(v, u, beta, ages, idx):
    """Apply each demo's factors to v (B n d) in the order idx (n K); frozen after age_j steps."""
    ages_t = torch.as_tensor(ages, device=v.device)
    idx = idx.to(v.device)
    ar = torch.arange(v.shape[0], device=v.device)[:, None]
    for i in range(int(ages_t.max())):
        k = idx[:, i]                                                # n
        new = reflect(v, u[ar, k[None]], beta[ar, k[None]])          # u[ar, k]: B n d
        v = torch.where((i < ages_t)[None, :, None], new, v)
    return v


class ErrorPath(nn.Module):
    def __init__(self, variant, n_marks, beta_mode='two'):
        super().__init__()
        self.variant, self.beta_mode = variant, beta_mode
        self.scale = nn.Parameter(torch.tensor(0.3))
        if variant != 'A':
            self.lag = mlp(5 + 2 * n_marks, 1)
        if variant not in ('A', 'B'):
            self.phi = nn.Linear(H, D_EMB, bias=False)
            self.qn, self.kn = nn.RMSNorm(D_EMB), nn.RMSNorm(D_EMB)
        if variant == 'End':
            self.end = mlp(4 * WIN, 1)
        if variant in PATH_VARIANTS:
            self.trans = mlp(2 * WIN, D_EMB + 1)
        if variant == 'Cmean':
            self.pool = mlp(2 * WIN, D_EMB)

    def transitions(self, b):
        src = b['er'] if self.variant == 'Cerr' else b['yr']
        z = self.trans(window_features(src))                         # B K d+1
        u = F.normalize(z[..., :-1], dim=-1)
        beta = torch.full_like(z[..., -1], 2.0) if self.beta_mode == 'two' else 2 * torch.sigmoid(z[..., -1])
        return u, beta

    def path_scores(self, fq, fk, ages, u, beta):
        """C: fq^T P_j fk_j (single backward sweep of fq: P_j^T fq recorded after k = age_j - 1).
        Crev: fq^T P_j^T fk_j.  Also returns |P_j^T fq| / |fq|."""
        n = len(ages)
        stop = [int(a) - 1 for a in ages]
        if self.variant != 'Crev':
            w, rec, pos = fq, [None] * n, 0
            order = sorted(range(n), key=lambda j: stop[j])
            for k in range(stop[order[-1]] + 1):
                w = reflect(w, u[:, k], beta[:, k])
                while pos < n and stop[order[pos]] == k:
                    rec[order[pos]] = w
                    pos += 1
            wq = torch.stack(rec, 1)                                 # B n d
            return (wq * fk).sum(-1), wq.norm(dim=-1) / fq.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v = sweep(fk, u, beta, ages, factor_order(ages, 'PT'))
        return (fq[:, None] * v).sum(-1), v.norm(dim=-1) / fk.norm(dim=-1).clamp_min(1e-8)

    def embed(self, b):
        return self.qn(self.phi(b['q'])), self.kn(self.phi(b['K']))

    def forward(self, b, order=None):
        V = b['V']
        diag = {}
        if self.variant == 'A':
            score = V.new_zeros(V.shape[:2])
        else:
            score = self.lag(lag_features(b['ages'].to(V.device), b['mt'], b['ms']))[..., 0]
        if self.variant in ('Bp', 'End'):
            fq, fk = self.embed(b)
            score = score + (fq[:, None] * fk).sum(-1) / math.sqrt(D_EMB)
        if self.variant == 'End':
            yt = window_features(b['yt'])[:, None].expand(-1, V.shape[1], -1)
            score = score + self.end(torch.cat([yt, window_features(b['ys'])], -1))[..., 0]
        if self.variant == 'Cmean':
            fq, fk = self.embed(b)
            z = self.pool(window_features(b['yr']))                  # B K d
            csum = z.cumsum(1)
            ages = b['ages']
            m = csum[:, ages.to(V.device) - 1] / ages.to(V.device, V.dtype)[None, :, None]   # B n d
            score = score + (fq[:, None] * 2 * torch.sigmoid(m) * fk).sum(-1) / math.sqrt(D_EMB)
        if self.variant in PATH_VARIANTS:
            fq, fk = self.embed(b)
            u, beta = self.transitions(b)
            if order is None:
                ps, qratio = self.path_scores(fq, fk, b['ages'], u, beta)
            else:                                                    # diagnostic factor order on keys
                ps = (fq[:, None] * sweep(fk, u, beta, b['ages'], order)).sum(-1)
                qratio = None
            score = score + ps / math.sqrt(D_EMB)
            diag = dict(beta=beta.detach(), qratio=None if qratio is None else qratio.detach())
        att = torch.softmax(score, -1)
        diag['entropy'] = (-(att * att.clamp_min(1e-12).log()).sum(-1) / math.log(att.shape[-1])).detach()
        diag['score_std'] = score.std(-1).detach()
        return self.scale * (att[..., None] * V).sum(1), diag


# ----------------------------------------------------------------------------- train / eval
def to_dev(b, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def all_cases(data, max_cases=None, seed=0):
    C = data['X'].shape[1]
    t_all = data['origins'].repeat_interleave(C)
    c_all = torch.arange(C).repeat(len(data['origins']))
    if max_cases is not None and len(t_all) > max_cases:
        idx = torch.randperm(len(t_all), generator=torch.Generator().manual_seed(seed))[:max_cases]
        t_all, c_all = t_all[idx], c_all[idx]
    return t_all, c_all


def mse_eval(model, data, ages, device, bs, max_cases=None, seed=0, order=None, diag_batches=20):
    """Squared errors (backbone, corrected) over (origin, channel) cases, plus path diagnostics."""
    t_all, c_all = all_cases(data, max_cases, seed)
    need = model.variant in PATH_VARIANTS + ('Cmean',)
    se_base = se = 0.0
    lead_base = torch.zeros(H, dtype=torch.float64)
    lead = torch.zeros(H, dtype=torch.float64)
    D = {k: [] for k in ('beta', 'qratio', 'kratio', 'entropy', 'score_std')}
    model.eval()
    with torch.no_grad():
        for bi, i in enumerate(range(0, len(t_all), bs)):
            b = to_dev(gather(data, t_all[i:i + bs], c_all[i:i + bs], ages, need), device)
            corr, diag = model(b, order)
            r0 = b['target'].double()
            r1 = (b['target'] - corr).double()
            se_base += float(r0.square().sum())
            se += float(r1.square().sum())
            lead_base += r0.square().sum(0).cpu()
            lead += r1.square().sum(0).cpu()
            if bi < diag_batches:
                D['entropy'].append(diag['entropy'].cpu())
                D['score_std'].append(diag['score_std'].cpu())
                if model.variant in PATH_VARIANTS and order is None:
                    D['beta'].append(diag['beta'].flatten()[::97].cpu())
                    if diag['qratio'] is not None:
                        D['qratio'].append(diag['qratio'].cpu())
                    fq, fk = model.embed(b)
                    u, beta = model.transitions(b)
                    pk = sweep(fk, u, beta, b['ages'], factor_order(b['ages'], 'P'))
                    D['kratio'].append((pk.norm(dim=-1) / fk.norm(dim=-1).clamp_min(1e-8)).cpu())
    n = len(t_all) * H
    out = dict(cases=len(t_all), mse_base=se_base / n, mse=se / n,
               lead_gain_pct=(100 * (lead_base - lead) / lead_base).tolist())
    qs = torch.tensor([0.1, 0.5, 0.9])
    out['attn_entropy_q10_50_90'] = torch.quantile(torch.cat(D['entropy']).float(), qs).tolist()
    out['score_std_q10_50_90'] = torch.quantile(torch.cat(D['score_std']).float(), qs).tolist()
    if D['beta']:
        out['beta_q10_50_90'] = torch.quantile(torch.cat(D['beta']).float(), qs).tolist()
        kr = torch.cat(D['kratio']).float()
        out['Pk_over_k_median_by_demo'] = kr.median(0).values.tolist()          # demos sorted by age
        if D['qratio']:
            key = 'PTk_over_k_median_by_demo' if model.variant == 'Crev' else 'PTq_over_q_median_by_demo'
            out[key] = torch.cat(D['qratio']).float().median(0).values.tolist()
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
    rstats = {s: residual_stats(d) for s, d in splits.items()}
    print(f'[residuals] {json.dumps(rstats)}', flush=True)
    del exp
    torch.cuda.empty_cache()
    model = ErrorPath(variant, splits['train']['M'].shape[1], args.beta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr = splits['train']
    C = tr['X'].shape[1]
    gen = torch.Generator().manual_seed(seed + 7)
    need = variant in PATH_VARIANTS + ('Cmean',)
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
        val = mse_eval(model, splits['val'], ages, device, args.eval_bs, args.val_cases, seed, diag_batches=1)
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
    t1 = time.time()
    test = mse_eval(model, splits['test'], ages, device, args.eval_bs, args.test_cases, seed)
    test['gain_pct'] = 100 * (test['mse_base'] - test['mse']) / test['mse_base']
    test['eval_sec'] = time.time() - t1
    print(f'[test] {variant} {kind} mse={test["mse"]:.6f} base={test["mse_base"]:.6f} gain={test["gain_pct"]:+.2f}% '
          f'entropy={test["attn_entropy_q10_50_90"]}', flush=True)
    if 'beta_q10_50_90' in test:
        print(f'[path] beta={test["beta_q10_50_90"]} |Pk|/|k| by demo={test["Pk_over_k_median_by_demo"]}', flush=True)
    order_diag = None
    if variant == 'C':
        # Diagnostic (not causal proof): same trained model, each demo's factors applied in a random order.
        sub = dict(model_order=mse_eval(model, splits['test'], ages, device, args.eval_bs, args.diag_cases, seed + 1,
                                        order=factor_order(ages, 'P'), diag_batches=1))
        perm = factor_order(ages, 'perm', torch.Generator().manual_seed(seed + 11))
        sub['permuted'] = mse_eval(model, splits['test'], ages, device, args.eval_bs, args.diag_cases, seed + 1,
                                   order=perm, diag_batches=1)
        order_diag = {k: dict(mse=v['mse'], mse_base=v['mse_base']) for k, v in sub.items()}
        print(f'[order diag] {order_diag}', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, demos=kind, variant=variant, beta_mode=args.beta, ages=ages,
               d=D_EMB, win=WIN, checkpoint=str(checkpoint), residual_stats=rstats,
               train_cases_per_epoch=args.iters * args.bs, args=vars(args), log=log, best_val=best, test=test,
               order_diag=order_diag)
    output = Path(args.out_dir) / f'{code}_H{H}_{kind}_{variant}_{args.beta}_s{seed}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T', choices=CELLS)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--demos', default='phase', choices=('dense', 'phase'))
    p.add_argument('--variant', required=True, choices=VARIANTS)
    p.add_argument('--beta', default='two', choices=('two', 'learn'))
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--iters', type=int, default=400)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--eval_bs', type=int, default=2048)
    p.add_argument('--val_cases', type=int, default=40000)
    p.add_argument('--test_cases', type=int, default=None)
    p.add_argument('--diag_cases', type=int, default=200000)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a.demos, a.variant, a)


if __name__ == '__main__':
    main()
