"""PAT-315 upper-bound diagnostic: how much of the backbone's error is linearly predictable from causal error info?

Case (origin t, channel c), target E_t[h] = y_{t+h} - f_t(y_{t-L:t})[h], h = 0..H-1.  At origin t only y_{<t} is
observed, so origin t-k has its first k leads observed (lead h <= k-1).  Features (all causal):
  fresh  own channel: 1-step errors E_{t-j}[0], j = 1..NJ (error series on y_{t-j}), and the errors on y_{t-1}
         from older origins E_{t-k}[k-1], k = 2..NK, plus a bias
  ret    the retrieval prediction for lead h: 14 same-phase demos, softmax(tau_e cos(E_{t-H}, E_{s-H})
         + tau_y cos(y96_t, y96_s)) . E_s  (tau of eval_raw_multistep; its scale is absorbed by the ridge)
  cross  all channels' 1-step errors E_{t-j}[0, :], j = 1..NX, projected on the top-R PCA directions of the
         train 1-step error vectors
One ridge per lead h (features shared over channels), fitted on the full train split; the relative ridge strength
is selected on val; test once.  Arms: ret | fresh | fresh+ret | fresh+cross | fresh+ret+cross.
All sums are second moments, so val/test MSE for any weights follow from the accumulated Gram statistics.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_error_path import CELLS, H, L, all_cases, demo_ages, load_split

NJ, NK, NX, R = 24, 24, 4, 16
TAU_E, TAU_Y, YL = 1.0, 4.0, 96
LAMS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SEGS = {'h1-24': (0, 24), 'h25-96': (24, 96), f'h97-{H}': (96, H)}
D_FRESH = 1 + NJ + (NK - 1)
D_CROSS = NX * R
ARMS = {'ret': ('ret',), 'fresh': ('fresh',), 'fresh+ret': ('fresh', 'ret'), 'fresh+cross': ('fresh', 'cross'),
        'fresh+ret+cross': ('fresh', 'ret', 'cross')}


def cosm(a, b):
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-1, -2)


def pca_dirs(data, device):
    e1 = data['E'][:, 0, :].to(device).double()                       # rows x C, 1-step errors
    e1 = e1 - e1.mean(0, keepdim=True)
    w, v = torch.linalg.eigh(e1.T @ e1 / len(e1))
    return v[:, -R:].flip(-1).float(), float(w[-R:].sum() / w.sum())


@torch.no_grad()
def stats(data, ages, P, device, bs):
    """Shared features S (B, D) = [fresh, cross]; per-lead feature r (B, H) = ret.
    Returns G (D,D), Sr (D,H), rr (H), ST (D,H), rT (H), TT (H), n_cases."""
    E, X = data['E'], data['X']
    ag = torch.as_tensor(ages)
    k = torch.arange(YL)
    j = torch.arange(1, NJ + 1)
    kk = torch.arange(2, NK + 1)
    jx = torch.arange(1, NX + 1)
    D = D_FRESH + D_CROSS
    acc = dict(G=torch.zeros(D, D), Sr=torch.zeros(D, H), rr=torch.zeros(H), ST=torch.zeros(D, H), rT=torch.zeros(H),
               TT=torch.zeros(H))
    acc = {a: b.double().to(device) for a, b in acc.items()}
    t_all, c_all = all_cases(data)
    for i in range(0, len(t_all), bs):
        t, c = t_all[i:i + bs], c_all[i:i + bs]
        s = t[:, None] - ag[None]
        cc = c[:, None]
        target = E[t - L, :, c].to(device).double()
        q, K, V = E[t - H - L, :, c].to(device), E[s - H - L, :, cc].to(device), E[s - L, :, cc].to(device).double()
        yq = X[(t - 1)[:, None] - k, c[:, None]].to(device)
        ys = X[(s - 1)[..., None] - k, cc[..., None]].to(device)
        a = torch.softmax((TAU_E * cosm(q[:, None], K) + TAU_Y * cosm(yq[:, None], ys))[:, 0].double(), -1)
        ret = (a[..., None] * V).sum(1)                                 # B H
        f1 = E[t[:, None] - j - L, 0, cc]                               # B NJ   E_{t-j}[0]
        f2 = E[t[:, None] - kk - L, kk - 1, cc]                         # B NK-1 E_{t-k}[k-1] (error on y_{t-1})
        ex = E[t[:, None] - jx - L, 0, :].to(device)                    # B NX C
        z = (ex @ P).flatten(1)                                         # B NX*R
        S = torch.cat([torch.ones(len(t), 1), f1, f2], 1).to(device)
        S = torch.cat([S, z], 1).double()
        acc['G'] += S.T @ S
        acc['Sr'] += S.T @ ret
        acc['rr'] += ret.square().sum(0)
        acc['ST'] += S.T @ target
        acc['rT'] += (ret * target).sum(0)
        acc['TT'] += target.square().sum(0)
    out = {a: b.cpu() for a, b in acc.items()}
    out['n'] = len(t_all)
    return out


def arm_index(arm):
    idx = []
    if 'fresh' in ARMS[arm]:
        idx += list(range(D_FRESH))
    if 'cross' in ARMS[arm]:
        idx += list(range(D_FRESH, D_FRESH + D_CROSS))
    if 'fresh' not in ARMS[arm]:
        idx = [0] + idx                                                 # keep the bias
    return idx, 'ret' in ARMS[arm]


def lead_system(st, h, idx, use_ret):
    """Gram A, moment b, target energy for lead h on the arm's features."""
    i = torch.tensor(idx)
    A = st['G'][i][:, i]
    b = st['ST'][i, h]
    if use_ret:
        A = torch.cat([torch.cat([A, st['Sr'][i, h:h + 1]], 1),
                       torch.cat([st['Sr'][i, h], st['rr'][h:h + 1]])[None]], 0)
        b = torch.cat([b, st['rT'][h:h + 1]])
    return A, b, st['TT'][h]


def fit(st, idx, use_ret, lam):
    W = []
    for h in range(H):
        A, b, _ = lead_system(st, h, idx, use_ret)
        reg = lam * A.diagonal().mean() * torch.eye(len(A), dtype=A.dtype)
        W.append(torch.linalg.solve(A + reg, b))
    return W


def sse(st, idx, use_ret, W):
    """Per-lead residual energy sum (T - x.w)^2 = TT - 2 w.b + w.A.w."""
    out = []
    for h in range(H):
        A, b, tt = lead_system(st, h, idx, use_ret)
        w = W[h]
        out.append(float(tt - 2 * w @ b + w @ A @ w))
    return torch.tensor(out, dtype=torch.float64)


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    device = exp.device
    cell = CELLS[code]
    ages = demo_ages('phase', cell['history'], cell['period'], cell['stride'])
    split = {sp: load_split(exp, sp, cell['history'], args.forecast_bs) for sp in ('train', 'val', 'test')}
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    test_base = float(split['test']['E'][split['test']['origins'] - L].double().square().mean())
    if abs(test_base - official) > 1e-4:
        raise AssertionError(f'base cached={test_base} official={official}')
    del exp
    torch.cuda.empty_cache()
    P, pca_frac = pca_dirs(split['train'], device)
    st = {}
    for sp in ('train', 'val', 'test'):
        st[sp] = stats(split[sp], ages, P, device, args.bs)
        split[sp] = None
        print(f'[stats] {sp} cases {st[sp]["n"]}', flush=True)
    base_h = st['test']['TT'] / st['test']['n']
    assert abs(float(base_h.mean()) - test_base) < 1e-6, (float(base_h.mean()), test_base)
    result = {}
    for arm in ARMS:
        idx, use_ret = arm_index(arm)
        fits = {lam: fit(st['train'], idx, use_ret, lam) for lam in LAMS}
        val = {lam: float(sse(st['val'], idx, use_ret, W).sum()) for lam, W in fits.items()}
        lam = min(val, key=val.get)
        e = sse(st['test'], idx, use_ret, fits[lam]) / st['test']['n']   # per-lead test MSE
        seg = {name: 100 * float(1 - e[a:b].sum() / base_h[a:b].sum()) for name, (a, b) in SEGS.items()}
        result[arm] = dict(lam=lam, n_feat=len(idx) + int(use_ret), test_mse=float(e.mean()),
                           test_gain_pct=100 * float(1 - e.mean() / base_h.mean()), seg_gain_pct=seg,
                           val_mse=val[lam] / (st['val']['n'] * H))
        print(f'[test] {arm:16s} lam={lam:g} gain={result[arm]["test_gain_pct"]:+.3f}% '
              + ' '.join(f'{k}:{v:+.2f}' for k, v in seg.items()), flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, ages=ages, NJ=NJ, NK=NK, NX=NX, R=R, pca_frac=pca_frac,
               tau_e=TAU_E, tau_y=TAU_Y, lams=LAMS, checkpoint=str(checkpoint), test_base=test_base,
               base_seg={name: float(base_h[a:b].mean()) for name, (a, b) in SEGS.items()}, arms=result)
    output = Path(args.out_dir) / f'{code}_H{H}_L{L}_freshdiag_s{seed}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T', choices=CELLS)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--bs', type=int, default=4096)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
