"""Diagnose the W1-step blow-up of eval_itransformer_tefl_twolayer.py.

For every (origin, channel) case and the val-chosen avg+W2+W1 steps of each cell,
record the three quantities
  |K|_F                      (demo keys, n x L; its square is the W1-step denominator)
  |q|_2                      (query key)
  a1 |dW1 q| / (|W1 q| + eps) (relative pre-activation shift of the W1 step)
and per-case MSE with the W1 step, without it (same a_b, a2, a1 = 0) and of TEFL.
A case is divergent when MSE(with W1) > 10 x MSE(without W1).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_itransformer_tefl_icl import forecast_errors, load_model
from patchmem.eval_itransformer_tefl_avg_icl import PERIODS, history_for, phase_ages

L, H = 96, 192
EPS = 1e-6
DIV = 10.0
QS = (0.01, 0.1, 0.5, 0.9, 0.99, 1.0)


@torch.inference_mode()
def diag_chunk(q, K, V, target, W1, W2, ab, a2, a1):
    """Per (b, C): |K|_F, |q|, relative W1 shift, MSE with W1, without W1, TEFL."""
    pre_q, pre_k = q @ W1.T, K @ W1.T
    hq, hk = torch.relu(pre_q), torch.relu(pre_k)
    mask = (pre_k > 0).to(K.dtype)
    r = V - hk @ W2.T
    b = ab * r.mean(-2)
    v = r - b[:, :, None]
    G = hk @ hk.transpose(-1, -2)
    kq = (K * q[:, :, None]).sum(-1)
    ksq = K.square().sum((-1, -2)).clamp_min(1e-12)
    den = hq.norm(dim=-1) * hk.norm(dim=-1).sum(-1) + 1e-6
    eta = (a2 / den)[..., None, None]
    e = v - eta * (G @ v)
    eW2 = e @ W2 + eta * ((e @ v.transpose(-1, -2)) @ hk)
    dq = ((eW2 * mask) * kq[..., None]).sum(-2) / ksq[..., None]

    def mse(hq_):
        sim = (hk * hq_[:, :, None]).sum(-1)
        pred = hq_ @ W2.T + eta[..., 0] * (sim[..., None] * v).sum(-2) + b
        return (target - pred).square().mean(-1)

    return dict(K_F=K.square().sum((-1, -2)).sqrt(), q_norm=q.norm(dim=-1),
                rel_shift=a1 * dq.norm(dim=-1) / (pre_q.norm(dim=-1) + EPS),
                mse_w1=mse(torch.relu(pre_q + a1 * dq)), mse_now1=mse(hq),
                mse_tefl=(target - hq @ W2.T).square().mean(-1))


@torch.inference_mode()
def diag_cell(E, origins, W1, W2, ages, chunk, steps):
    ages = torch.as_tensor(ages, dtype=torch.long)
    dev = W1.device
    out = {}
    for start in range(0, len(origins), chunk):
        t = origins[start:start + chunk]
        s = t[:, None] - ages[None]
        q = E[t - H - L].permute(0, 2, 1).to(device=dev, dtype=W1.dtype)
        target = E[t - L].permute(0, 2, 1).to(device=dev, dtype=W1.dtype)
        K = E[s - H - L].permute(0, 3, 1, 2).to(device=dev, dtype=W1.dtype)
        V = E[s - L].permute(0, 3, 1, 2).to(device=dev, dtype=W1.dtype)
        for k, x in diag_chunk(q, K, V, target, W1, W2, *steps).items():
            out.setdefault(k, []).append(x.cpu())
    return {k: torch.cat(v) for k, v in out.items()}       # each: n_origins x C


def quantiles(x):
    x = x.flatten()
    if x.numel() == 0:
        return None
    return {str(p): float(torch.quantile(x, p)) if x.numel() < 2 ** 24 else float(x.sort().values[int(p * (x.numel() - 1))])
            for p in QS}


def summarize(d, origins, top):
    div = d['mse_w1'] > DIV * d['mse_now1']
    per_ch = div.sum(0)
    res = dict(n_cases=int(div.numel()), n_div=int(div.sum()),
               mse_w1=float(d['mse_w1'].mean()), mse_now1=float(d['mse_now1'].mean()),
               mse_tefl=float(d['mse_tefl'].mean()),
               excess_from_div=float((d['mse_w1'] - d['mse_now1'])[div].sum() / (d['mse_w1'] - d['mse_now1']).sum()),
               n_channels_with_div=int((per_ch > 0).sum()),
               top_div_channels={int(c): int(per_ch[c]) for c in per_ch.argsort(descending=True)[:20].tolist() if per_ch[c] > 0})
    for k in ('K_F', 'q_norm', 'rel_shift'):
        res[k] = dict(div=quantiles(d[k][div]), rest=quantiles(d[k][~div]))
    flat = d['mse_w1'].flatten().argsort(descending=True)[:top]
    C = d['mse_w1'].shape[1]
    res['worst'] = [dict(origin_idx=int(i // C), origin=int(origins[i // C]), channel=int(i % C),
                         **{k: float(d[k].flatten()[i]) for k in d}) for i in flat.tolist()]
    return res


def run(code, seed, chunk, forecast_bs, top, src, output, baseline_tag):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chosen = json.loads(src.read_text())['cells']
    exp, _ = load_model(code, seed, device, baseline_tag)
    W1 = exp.model.W1.weight.detach().to(device=device, dtype=torch.float64)
    W2 = exp.model.W2.weight.detach().to(device=device, dtype=torch.float64)
    cells = {f'{ph}_n{n}': phase_ages(P, n) for ph, P in zip(('day', 'week'), PERIODS[code]) for n in (4, 8, 16)}
    history = max(history_for(a) for a in cells.values())
    result = {}
    for split in ('val', 'test'):
        E, origins, _ = forecast_errors(exp, split, history, forecast_bs)
        for cell, ages in cells.items():
            steps = tuple(chosen[cell]['families']['avg+W2+W1']['steps'])
            s = summarize(diag_cell(E, origins, W1, W2, ages, chunk, steps), origins, top)
            if split == 'test':
                ref = chosen[cell]['families']['avg+W2+W1']['test_mse']
                assert abs(s["mse_w1"] - ref) <= 1e-3 * max(1.0, abs(ref)), (cell, s['mse_w1'], ref)
            result.setdefault(cell, dict(steps=steps))[split] = s
            print(f'[{split}] {cell:9s} steps={steps} div={s["n_div"]}/{s["n_cases"]} '
                  f'mse w1={s["mse_w1"]:.4g} now1={s["mse_now1"]:.4g} excess_from_div={s["excess_from_div"]:.3f} | '
                  + ' '.join(f'{k} div50={(s[k]["div"] or {}).get("0.5", float("nan")):.3g} '
                             f'rest50={(s[k]["rest"] or {}).get("0.5", float("nan")):.3g}' for k in ('K_F', 'q_norm', 'rel_shift')), flush=True)
        del E
    out = dict(dataset=code, seed=seed, div_rule=f'mse_w1 > {DIV} x mse_now1', src=str(src), cells=result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, choices=PERIODS)
    parser.add_argument('--seed', type=int, default=3047)
    parser.add_argument('--chunk', type=int, default=4)
    parser.add_argument('--forecast_bs', type=int, default=8)
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--baseline_tag', default='TEFL_paper_L96_E192')
    args = parser.parse_args()
    src = Path(f'patchmem/logs/server/itr_twolayer/{args.data}_s{args.seed}.json')
    output = Path(f'patchmem/logs/server/itr_twolayer_diag/{args.data}_s{args.seed}.json')
    run(args.data, args.seed, args.chunk, args.forecast_bs, args.top, src, output, args.baseline_tag)


if __name__ == '__main__':
    main()
