"""PAT-315: what does the iTransformer-TEFL map g(eps) = W2 relu(W1 eps) learn?

Per test origin t and channel c (TEFL_paper_L96_E192 checkpoints, L 96, H 192):
  eps = E[t - H]   f-only error vector of the forecast issued one horizon ago (targets t-H .. t-1), the TEFL input
  e   = E[t]       f-only error of the current forecast (targets t .. t+H-1), g = TEFL's correction, y^ = f + g
TEFL gain = sum e^2 - sum (e - g)^2.  Simple structures z[h] (one coefficient alpha_h per lead, fitted on val):
  diag      eps[h]                              same lead, lag H
  mean      mean_j eps[j]                       level shift (rank 1)
  phase     mean of eps[j] over j == h + H (mod P)    folded daily template, P = daily period of the dataset
  last      eps[j*], j* the latest j with j == h + H (mod P)   most recent same-phase error
  week      eps[h + 24] (h < 168) else eps[h - 144]        same hour of week (hourly data, E / T only)
  amp       f[h] - mean_j f[j]                  amplitude of the current forecast (g cannot see f; tests correlation)
  lin       A eps + b (192 x 192), and its SVD truncations rank 1 / 2 / 4
Two fits per structure: mimic (alpha = argmin |g - alpha z|^2 on val) and direct (argmin |e - alpha z|^2 on val).
Reported on test: R^2 of the mimic against g, and retained = (sum e^2 - sum (e - alpha z)^2) / TEFL gain.
Criterion (fixed): TEFL "is" a simple feature if one of diag / mean / phase / last / week / amp (mimic) retains
>= 80% of the TEFL gain on 3/3 seeds.  Also: mean Jacobian W2 diag(rate) W1, ReLU switching statistics, per lead
std(g) / std(eps) and corr(g, eps[h]), per target phase mean g, per channel gain share.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
np.Inf = np.inf
from data_provider.data_factory import data_provider
from patchmem.eval_itransformer_tefl_icl import load_model

L, H = 96, 192
PERIOD = {'E': 24, 'M1': 96, 'M2': 96, 'W': 144, 'T': 24}
HOURLY = ('E', 'T')
TAG = 'TEFL_paper_L96_E192'
RANKS = (1, 2, 4)


@torch.inference_mode()
def cache(exp, split, bs):
    """E (n H C) f-only errors and F (n H C) forecasts for every origin of the split prefix; official origins."""
    args = exp.args
    args.seq_len = L + H
    ds, _ = data_provider(args, split)
    X = torch.as_tensor(np.asarray(ds.data_x), dtype=torch.float32)
    mk = torch.as_tensor(np.asarray(ds.data_stamp), dtype=torch.float32)
    origins = torch.arange(L, len(X) - H + 1)
    E = torch.empty((len(origins), H, X.shape[1]))
    F = torch.empty_like(E)
    off = torch.arange(-L, 0)
    for i in range(0, len(origins), bs):
        o = origins[i:i + bs]
        idx = o[:, None] + off[None]
        p = exp.model.base(X[idx].to(exp.device), mk[idx].to(exp.device), None, None).cpu()
        F[i:i + len(o)] = p
        E[i:i + len(o)] = X[o[:, None] + torch.arange(H)] - p
    off_o = torch.arange(L + H, L + H + len(ds))
    assert off_o[-1] + H == len(X)
    return E, F, off_o, X.shape[1]


def proxy_index(code):
    """Index maps j(h) into eps for the gather proxies."""
    P = PERIOD[code]
    h = torch.arange(H)
    r = (h + H) % P
    last = (H - 1) - ((H - 1 - r) % P)
    assert bool(((last - h - H) % P == 0).all()) and bool((last <= H - 1).all()) and bool((last >= H - P).all())
    phase_masks = torch.stack([(torch.arange(H) % P) == (int(x) % P) for x in r]).double()    # H(h) x H(j)
    phase_masks = phase_masks / phase_masks.sum(1, keepdim=True)
    idx = dict(last=last)
    if code in HOURLY:
        week = torch.where(h < 168, h + 24, h - 144)
        # target time t + h; eps[j] is at time t - H + j; same hour of week <=> (j - H - h) % 168 == 0
        assert bool(((week - H - h) % 168 == 0).all())
        idx['week'] = week
    return idx, phase_masks


def proxies(eps, f, idx, pm):
    """eps, f (B H) -> dict name -> z (B H)."""
    z = dict(diag=eps, mean=eps.mean(1, keepdim=True).expand_as(eps), phase=eps @ pm.T,
             amp=f - f.mean(1, keepdim=True))
    for k, v in idx.items():
        z[k] = eps[:, v]
    return z


def run(code, seed, args):
    dev = torch.device('cuda')
    exp, ckpt = load_model(code, seed, dev, TAG)
    W1 = exp.model.W1.weight.detach().double()                                  # R x H
    W2 = exp.model.W2.weight.detach().double()                                  # H x R
    met = json.loads((ckpt.parents[1].parent / 'results' / ckpt.parent.name / 'tefl_metrics.json').read_text())
    idx, pm = proxy_index(code)
    idx = {k: v.to(dev) for k, v in idx.items()}
    pm = pm.to(dev)
    names = ['diag', 'mean', 'phase', 'last', 'amp'] + (['week'] if code in HOURLY else [])
    R = W1.shape[0]
    mom = {}
    for split in ('val', 'test'):
        E, F, orig, C = cache(exp, split, args.bs)
        n0 = L                                                                    # E index of origin o is o - L
        acc = dict(ee=torch.zeros(H, dtype=torch.float64, device=dev), gg=torch.zeros(H, dtype=torch.float64, device=dev),
                   eg=torch.zeros(H, dtype=torch.float64, device=dev), g1=torch.zeros(H, dtype=torch.float64, device=dev),
                   epsg=torch.zeros(H, dtype=torch.float64, device=dev), eps1=torch.zeros(H, dtype=torch.float64, device=dev), eps2=torch.zeros(H, dtype=torch.float64, device=dev),
                   n=0, xx=torch.zeros(H + 1, H + 1, dtype=torch.float64, device=dev),
                   xg=torch.zeros(H + 1, H, dtype=torch.float64, device=dev),
                   xe=torch.zeros(H + 1, H, dtype=torch.float64, device=dev),
                   rate=torch.zeros(R, dtype=torch.float64, device=dev),
                   chan_ee=torch.zeros(C, dtype=torch.float64, device=dev),
                   chan_r=torch.zeros(C, dtype=torch.float64, device=dev))
        P = PERIOD[code]
        acc['ph_g'] = torch.zeros(P, dtype=torch.float64, device=dev)
        acc['ph_e'] = torch.zeros(P, dtype=torch.float64, device=dev)
        acc['ph_n'] = torch.zeros(P, dtype=torch.float64, device=dev)
        for k in names:
            acc[k] = {q: torch.zeros(H, dtype=torch.float64, device=dev) for q in ('zz', 'zg', 'ze')}
        for i in range(0, len(orig), args.chunk):
            t = orig[i:i + args.chunk]
            eps = E[t - H - n0].permute(0, 2, 1).to(dev).double()                 # B C H
            e = E[t - n0].permute(0, 2, 1).to(dev).double()
            f = F[t - n0].permute(0, 2, 1).to(dev).double()
            pre = eps @ W1.T
            g = torch.relu(pre) @ W2.T
            B = eps.shape[0]
            eps2, e2, g2, f2 = eps.reshape(-1, H), e.reshape(-1, H), g.reshape(-1, H), f.reshape(-1, H)
            acc['ee'] += e2.square().sum(0)
            acc['gg'] += g2.square().sum(0)
            acc['eg'] += (e2 * g2).sum(0)
            acc['g1'] += g2.sum(0)
            acc['epsg'] += (eps2 * g2).sum(0)
            acc['eps2'] += eps2.square().sum(0)
            acc['eps1'] += eps2.sum(0)
            acc['n'] += eps2.shape[0]
            x = torch.cat([eps2, torch.ones_like(eps2[:, :1])], 1)
            acc['xx'] += x.T @ x
            acc['xg'] += x.T @ g2
            acc['xe'] += x.T @ e2
            acc['rate'] += (pre > 0).double().reshape(-1, R).sum(0)
            acc['chan_ee'] += e.square().sum((0, 2))
            acc['chan_r'] += (e - g).square().sum((0, 2))
            ph = ((t[:, None] + torch.arange(H)) % P).to(dev)                     # B H target phase (relative to split)
            acc['ph_g'].index_add_(0, ph.flatten(), g.sum(1).flatten())
            acc['ph_e'].index_add_(0, ph.flatten(), e.sum(1).flatten())
            acc['ph_n'].index_add_(0, ph.flatten(), torch.full_like(ph.flatten(), C, dtype=torch.float64))
            z = proxies(eps2, f2, idx, pm)
            for k in names:
                acc[k]['zz'] += z[k].square().sum(0)
                acc[k]['zg'] += (z[k] * g2).sum(0)
                acc[k]['ze'] += (z[k] * e2).sum(0)
        mom[split] = acc
        print(f'[{split}] origins {len(orig)} channels {C}', flush=True)
        del E, F
    v, te = mom['val'], mom['test']
    n = te['n']
    base = float(te['ee'].sum()) / (n * H)
    tefl = float((te['ee'] - 2 * te['eg'] + te['gg']).sum()) / (n * H)
    if abs(tefl - met['mse']) > 1e-4 or abs(base - met['base_mse']) > 1e-4:
        raise AssertionError(f'mismatch tefl {tefl} vs {met["mse"]}, base {base} vs {met["base_mse"]}')
    gain = float((te['ee'] - (te['ee'] - 2 * te['eg'] + te['gg'])).sum())
    gvar = te['gg'] - te['g1'].square() / n                                       # per lead centered SS of g
    res = {}

    def score(name, pred_gz, pred_ee, gg_res):
        """pred_gz: sum over test of (alpha z) g; pred_ee: test sum (e - alpha z)^2 per lead; gg_res: sum (g - alpha z)^2."""
        return dict(r2=float(1 - gg_res.sum() / gvar.sum()),
                    retained=float((te['ee'].sum() - pred_ee.sum()) / gain))

    for k in names:
        for fit, key in (('mimic', 'zg'), ('direct', 'ze')):
            a = torch.where(v[k]['zz'] > 0, v[k][key] / v[k]['zz'].clamp_min(1e-30), torch.zeros_like(v[k]['zz']))
            sse_e = te['ee'] - 2 * a * te[k]['ze'] + a.square() * te[k]['zz']
            sse_g = te['gg'] - 2 * a * te[k]['zg'] + a.square() * te[k]['zz']
            res[f'{k}_{fit}'] = dict(**score(k, None, sse_e, sse_g), alpha_mean=float(a.mean()),
                                     alpha_by_bucket=a.view(8, -1).mean(1).tolist())
    eye = torch.eye(H + 1, dtype=torch.float64, device=dev) * 1e-8 * v['xx'].diagonal().mean()
    for fit, key in (('mimic', 'xg'), ('direct', 'xe')):
        Ab = torch.linalg.solve(v['xx'] + eye, v[key])                            # (H+1) x H, last row bias
        A, b = Ab[:H], Ab[H:]
        U, S, Vh = torch.linalg.svd(A)
        for r in (None,) + RANKS:
            Ar = A if r is None else (U[:, :r] * S[:r]) @ Vh[:r]
            M = torch.cat([Ar, b], 0)
            sse_e = te['ee'] - 2 * (M * te['xe']).sum(0) + (M * (te['xx'] @ M)).sum(0)
            sse_g = te['gg'] - 2 * (M * te['xg']).sum(0) + (M * (te['xx'] @ M)).sum(0)
            res[f'lin{"" if r is None else f"_r{r}"}_{fit}'] = score('lin', None, sse_e, sse_g)
        if fit == 'mimic':
            A_mimic, S_mimic = A, S
            sg = torch.sign(U[:, 0].sum())                                        # orient: input weights sum >= 0
            v1, u1 = U[:, 0] * sg, Vh[0] * sg * S[0]                              # eps @ A_r1 = (eps . v1) u1
            rank1 = dict(v1_in=v1.tolist(), u1_out=u1.tolist(), sv_share=float(S[0] ** 2 / S.square().sum()),
                         v1_cos_uniform=float(v1.sum() / (H ** 0.5)), u1_cos_uniform=float(u1.sum() / (u1.norm() * H ** 0.5)))
    rate = te['rate'] / (n)
    J = W2 @ torch.diag(rate) @ W1                                                # mean Jacobian dg/deps, H x H
    sw = ((rate > 0.05) & (rate < 0.95)).float().mean()
    gc = te['gg'] - te['g1'].square() / n                                          # centered per lead
    ec = te['eps2'] - te['eps1'].square() / n
    std_ratio = (gc / ec).sqrt()
    corr_diag = (te['epsg'] - te['eps1'] * te['g1'] / n) / (gc * ec).sqrt()
    chan_gain = te['chan_ee'] - te['chan_r']
    srt = chan_gain.sort(descending=True).values
    C = len(chan_gain)
    out = dict(dataset=code, seed=seed, checkpoint=str(ckpt), L=L, H=H, period=PERIOD[code], test_cells=n,
               base_mse=base, tefl_mse=tefl, tefl_gain_pct=100 * (base - tefl) / base, arms=res,
               relu=dict(always_on=float((rate >= 0.95).float().mean()), always_off=float((rate <= 0.05).float().mean()),
                         switching=float(sw), rate=rate.tolist()),
               jac=dict(diag_mean=float(J.diagonal().mean()), offdiag_abs_mean=float((J - torch.diag(J.diagonal())).abs().mean()),
                        sv=torch.linalg.svdvals(J)[:16].tolist(), lin_sv=S_mimic[:16].tolist()), rank1=rank1,
               lead=dict(std_ratio_bucket=std_ratio.view(8, -1).mean(1).tolist(),
                         corr_diag_bucket=corr_diag.view(8, -1).mean(1).tolist(),
                         g_mean_bucket=(te['g1'] / n).view(8, -1).mean(1).tolist()),
               phase=dict(g_mean=(te['ph_g'] / te['ph_n']).tolist(), e_mean=(te['ph_e'] / te['ph_n']).tolist()),
               channel=dict(pos_frac=float((chan_gain > 0).double().mean()),
                            top10_share=float(srt[:max(1, C // 10)].sum() / chan_gain.sum())))
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f'{code}_tefl_map_s{seed}.npz', J=J.cpu().numpy(), A=A_mimic.cpu().numpy(), rate=rate.cpu().numpy())
    p = od / f'{code}_tefl_map_s{seed}.json'
    p.write_text(json.dumps(out, indent=2) + '\n')
    for k, r in res.items():
        print(f'[test] {k:16s} r2 {r["r2"]:+.3f} retained {r["retained"]:+.3f}', flush=True)
    print(f'[done] {p} tefl gain {out["tefl_gain_pct"]:.2f}% switching {float(sw):.2f}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, choices=PERIOD)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--out_dir', default='patchmem/logs/server/tefl_map')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
