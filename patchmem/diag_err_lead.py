"""PAT-315: is the backbone's forecast error shared across the H leads, or a different kind of error per lead?

Frozen standalone iTransformer (L96), H from env ITR_H (192 / 720).  Error stream E[o, h, c] = y[o+h] - f(o)[h] over
train + val origins only (test is never streamed).  Three H-vectors per (origin, channel), each centred per (lead, channel):
  orig  e_o = E[o, :]                     one forecast's errors over its leads
  diag  d_t = E[t-1-l, l], l < H          all forecasts of the same target y[t-1] (the observation diagonal)
  pred  e^_o = W^T x_o                    the linearly predictable part: pooled ridge 'all' (eps, u, D; eval_err_value),
                                          lam = 0.1 fixed, fitted on train origins, evaluated on val origins
For each, the pooled H x H covariance S (over origins and channels) and per-channel covariances give
  eig share top1 / top3 / top10, participation ratio PR = (tr S)^2 / tr S^2,
  bucket retention R_nb = tr(P_nb S) / tr S, P_nb = projection on vectors constant inside nb equal lead buckets
  (nb = 1: share of a common shift over all leads; nb = 8: the decomp2 carry granularity), corr vs lead lag.
Decision rules (fixed before the runs, pooled S, 3/3 seeds):
  shared in H        R_1 >= 0.8              (one scalar per origin captures the error)
  low-dimensional    top3 >= 0.9
  8 buckets suffice  R_8 >= 0.9 for 'pred'   (bucket granularity keeps the actionable part)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import patchmem.eval_err_value as ev
from patchmem.eval_err_value import H

NBS = (1, 4, 8, 24, 48)
LAM = 0.1


def bucket_proj(nb, dev):
    b = torch.arange(H, device=dev) * nb // H                                         # equal split of the leads
    B = torch.nn.functional.one_hot(b, nb).double()
    return B @ torch.linalg.inv(B.T @ B) @ B.T


def summarize(S, dev):
    lam = torch.linalg.eigvalsh(S).flip(0).clamp_min(0)
    tr = lam.sum()
    sd = S.diagonal().clamp_min(1e-30).sqrt()
    R = S / sd[:, None] / sd[None]
    lag = torch.stack([R.diagonal(k).mean() for k in range(H)])
    return dict(top1=float(lam[0] / tr), top3=float(lam[:3].sum() / tr), top10=float(lam[:10].sum() / tr),
                pr=float(tr ** 2 / (lam ** 2).sum()),
                **{f'R{nb}': float(torch.trace(bucket_proj(nb, dev) @ S) / torch.trace(S)) for nb in NBS},
                corr_lag=lag.tolist(), var_lead=S.diagonal().tolist()), R


def per_channel(Sc, dev):
    out = {}
    lam = torch.linalg.eigvalsh(Sc).flip(1).clamp_min(0)                               # C H
    tr = lam.sum(1).clamp_min(1e-30)
    vals = dict(top1=lam[:, 0] / tr, top3=lam[:, :3].sum(1) / tr, pr=tr ** 2 / (lam ** 2).sum(1).clamp_min(1e-30))
    trS = Sc.diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-30)
    for nb in NBS:
        vals[f'R{nb}'] = torch.einsum('hg,cgh->c', bucket_proj(nb, dev), Sc) / trS
    for k, v in vals.items():
        q = torch.quantile(v.float().cpu(), torch.tensor([0.1, 0.5, 0.9]))
        out[k] = dict(q10=float(q[0]), median=float(q[1]), q90=float(q[2]))
    return out


def cov_stream(block, ks, dev, chunk, C):
    """Pooled and per-channel covariance of block(k) (B H C), centred per (lead, channel); two passes."""
    mu = torch.zeros(H, C, dtype=torch.float64, device=dev)
    n = 0
    for i in range(0, len(ks), chunk):
        x = block(ks[i:i + chunk])
        mu += x.sum(0)
        n += x.shape[0]
    mu /= n
    S = torch.zeros(H, H, dtype=torch.float64, device=dev)
    Sc = torch.zeros(C, H, H, dtype=torch.float64, device=dev)
    for i in range(0, len(ks), chunk):
        x = block(ks[i:i + chunk]) - mu
        S += torch.einsum('bhc,bgc->hg', x, x)
        xt = x.permute(2, 1, 0)                                                          # C H B
        Sc.baddbmm_(xt, xt.transpose(1, 2))                                              # in place: no C H H temp
    return S / (n * C), Sc / n


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    dev = exp.device
    X, M, n_tr, n_va = ev.load_series(exp, code)
    m_tr = args.m_tr if args.m_tr else n_tr
    assert m_tr >= 2 * H + 1
    o0 = max(ev.L, n_tr - m_tr)
    end = n_tr + n_va                                                                    # test is never streamed
    E = ev.stream_errors(exp, X[:end], M[:end], o0, args.bs)                             # (end - H + 1 - o0) H C
    C = X.shape[1]
    t_tr = torch.arange(o0 + H, n_tr - H + 1)
    t_va = torch.arange(n_tr, end - H + 1)
    t_all = torch.cat([t_tr, t_va])
    l = torch.arange(H)

    def orig(t):
        return E[t - o0].to(dev).double()

    def diag(t):
        k = t - o0
        return E[k[:, None] - 1 - l, l].to(dev).double()

    mo = ev.moments(E, t_tr, o0, dev, args.chunk)
    c = ev.cols('all').to(dev)
    W = ev.solve(mo, c, LAM)

    def pred(t):
        Xf, _ = ev.feats(E, t, o0, dev)
        return (Xf[:, :, c] @ W).permute(0, 2, 1)                                        # B H C

    def resid(t):
        Xf, Y = ev.feats(E, t, o0, dev)
        return (Y - Xf[:, :, c] @ W).permute(0, 2, 1)

    res, mats = {}, {}
    for name, blk, ts in (('orig', orig, t_all), ('diag', diag, t_all), ('pred', pred, t_va), ('resid', resid, t_va)):
        S, Sc = cov_stream(blk, ts, dev, args.chunk, C)
        res[name], R = summarize(S, dev)
        res[name]['per_channel'] = per_channel(Sc, dev)
        res[name]['n_origins'] = len(ts)
        mats[name] = R.cpu().numpy()
        r = res[name]
        print(f'[{name:5s}] top1 {r["top1"]:.3f} top3 {r["top3"]:.3f} PR {r["pr"]:.1f} R1 {r["R1"]:.3f} '
              f'R8 {r["R8"]:.3f} R24 {r["R24"]:.3f} | per-ch median R1 {r["per_channel"]["R1"]["median"]:.3f} '
              f'top3 {r["per_channel"]["top3"]["median"]:.3f}', flush=True)
        del Sc
    res['pred']['share_vs_resid'] = float(sum(res['pred']['var_lead']) /
                                          (sum(res['pred']['var_lead']) + sum(res['resid']['var_lead'])))
    out = dict(dataset=code, seed=seed, H=H, checkpoint=str(checkpoint), m_tr=m_tr, lam=LAM, nbs=NBS,
               rules=dict(shared=res['orig']['R1'] >= 0.8, low_dim=res['orig']['top3'] >= 0.9,
                          buckets8_ok=res['pred']['R8'] >= 0.9), stats=res)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    tag = f'{code}_H{H}_s{seed}'
    (od / f'{tag}.json').write_text(json.dumps(out, indent=2) + '\n')
    np.savez_compressed(od / f'{tag}.npz', **mats)
    print(f'[done] {od / tag} rules {out["rules"]}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, choices=('E', 'M1', 'M2', 'W', 'T', 'EL'))
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--m_tr', type=int, default=None)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--out_dir', default='patchmem/logs/server/err_lead')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
