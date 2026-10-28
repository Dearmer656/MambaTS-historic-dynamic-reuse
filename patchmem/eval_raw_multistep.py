"""PAT-315 training-free: K steps of kernel GD of the one-layer W1 model on one case's demos.

Key per origin o: cos on the error E_{o-H} and on the backbone's current input y[o-96:o], combined with the
train-selected temperatures of eval_raw_ylen (tau_e = 1, tau_y = 4).  a(x, x_j) = softmax_j over demos.
  f_0 = 0
  f_{k+1}(x) = f_k(x) + eta * sum_j a(x, x_j) (e_j - f_k(x_j))      e_j = E_{s_j} (demo value)
f is tracked at the query and at every demo (demo-demo weights from the same kernel).
  self : the demo row includes x_j itself (exact kernel GD)
  loo  : the demo row excludes x_j (each demo's residual is explained by the other demos)
K = 1 equals the one-step attention (e+y96 of eval_raw_ylen up to the scale eta*s).
K in {1,2,3,5,10}, eta and the output scale s are selected on a fixed 1M-case train subset; test once.
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
from patchmem.eval_raw_sim import S_GRID, mse_at

TAU_E, TAU_Y, YL = 1.0, 4.0, 96
KS = (1, 2, 3, 5, 10)
ETAS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
MODES = ('self', 'loo')


def cosm(a, b):
    """a: B n d, b: B m d -> B n m cosine."""
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-1, -2)


@torch.no_grad()
def sums(data, ages, device, bs, cands, max_cases=None, seed=0):
    """cands: set of (mode, eta, K); returns [sum t^2, sum t.f, sum f^2, n] for f = f_K(query)."""
    E, X = data['E'], data['X']
    ag = torch.as_tensor(ages)
    k = torch.arange(YL)
    t_all, c_all = all_cases(data, max_cases, seed)
    acc = {c: torch.zeros(3, dtype=torch.float64, device=device) for c in cands}
    n = 0
    kmax = max(c[2] for c in cands)
    for i in range(0, len(t_all), bs):
        t, c = t_all[i:i + bs], c_all[i:i + bs]
        s = t[:, None] - ag[None]
        cc = c[:, None]
        q = E[t - H - L, :, c].to(device)
        target = E[t - L, :, c].to(device).double()
        K = E[s - H - L, :, cc].to(device)
        V = E[s - L, :, cc].to(device).double()
        yq = X[(t - 1)[:, None] - k, c[:, None]].to(device)
        ys = X[(s - 1)[..., None] - k, cc[..., None]].to(device)
        lq = (TAU_E * cosm(q[:, None], K) + TAU_Y * cosm(yq[:, None], ys))[:, 0].double()     # B n
        ldd = (TAU_E * cosm(K, K) + TAU_Y * cosm(ys, ys)).double()                           # B n n
        aq = torch.softmax(lq, -1)
        tt = target.square().sum()
        n += target.numel()
        for mode in MODES:
            dd = ldd.clone()
            if mode == 'loo':
                dd.diagonal(dim1=-2, dim2=-1).fill_(float('-inf'))
            add = torch.softmax(dd, -1)
            for eta in sorted({c[1] for c in cands if c[0] == mode}):
                fq = torch.zeros_like(target)
                fd = torch.zeros_like(V)
                for step in range(1, kmax + 1):
                    r = V - fd
                    fq = fq + eta * (aq[..., None] * r).sum(1)
                    fd = fd + eta * add @ r
                    if (mode, eta, step) in acc:
                        acc[mode, eta, step] += torch.stack([tt, (target * fq).sum(), fq.square().sum()])
    return {c: torch.cat([a.cpu(), torch.tensor([float(n)], dtype=torch.float64)]) for c, a in acc.items()}


def run(code, seed, args):
    from patchmem.eval_itransformer_pure_icl import load_base
    exp, checkpoint, setting = load_base(code, seed)
    device = exp.device
    cell = CELLS[code]
    ages = demo_ages('phase', cell['history'], cell['period'], cell['stride'])
    sel = load_split(exp, 'train', cell['history'], args.forecast_bs)
    test = load_split(exp, 'test', cell['history'], args.forecast_bs)
    official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
    test_base = float(test['E'][test['origins'] - L].double().square().mean())
    if abs(test_base - official) > 1e-4:
        raise AssertionError(f'base cached={test_base} official={official}')
    del exp
    torch.cuda.empty_cache()
    cands = {(m, e, kk) for m in MODES for e in ETAS for kk in KS}
    acc_v = sums(sel, ages, device, args.bs, cands, args.select_cases, seed)
    del sel
    chosen = {}
    for m in MODES:
        for kk in KS:
            best = min(((e, sc) for e in ETAS for sc in S_GRID), key=lambda x: mse_at(acc_v[m, x[0], kk], x[1]))
            chosen[f'{m}:K{kk}'] = dict(mode=m, K=kk, eta=best[0], s=best[1], sel_mse=mse_at(acc_v[m, best[0], kk], best[1]))
        bestk = min((x for x in chosen.values() if x['mode'] == m), key=lambda x: x['sel_mse'])
        chosen[f'{m}:Kval'] = dict(bestk)
    acc_t = sums(test, ages, device, args.bs, {(x['mode'], x['eta'], x['K']) for x in chosen.values()})
    a0 = next(iter(acc_t.values()))
    base = float(a0[0] / a0[3])
    assert abs(base - test_base) < 1e-6, (base, test_base)
    result = {}
    for f, x in chosen.items():
        m = mse_at(acc_t[x['mode'], x['eta'], x['K']], x['s'])
        result[f] = dict(x, test_mse=m, test_gain_pct=100 * (base - m) / base)
        print(f'[test] {f:10s} K={x["K"]} eta={x["eta"]} s={x["s"]} gain={result[f]["test_gain_pct"]:+.3f}%', flush=True)
    out = dict(dataset=code, seed=seed, H=H, L=L, ages=ages, tau_e=TAU_E, tau_y=TAU_Y, y_len=YL, Ks=KS, etas=ETAS,
               select_split='train', select_cases=args.select_cases, checkpoint=str(checkpoint), test_base=base,
               families=result)
    output = Path(args.out_dir) / f'{code}_H{H}_L{L}_phase_multistep_s{seed}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='T', choices=CELLS)
    p.add_argument('--seed', type=int, default=3047)
    p.add_argument('--bs', type=int, default=2048)
    p.add_argument('--forecast_bs', type=int, default=8)
    p.add_argument('--select_cases', type=int, default=1000000)
    p.add_argument('--out_dir', default='patchmem/logs/server/error_path')
    a = p.parse_args()
    run(a.data, a.seed, a)


if __name__ == '__main__':
    main()
