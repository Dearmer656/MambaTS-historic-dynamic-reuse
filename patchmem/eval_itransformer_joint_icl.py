"""ICL-only (no TEFL) with one joint GD step on both layers from a random head.

Head g(eps) = W2 ReLU(W1 eps), W1 ~ U(+-1/sqrt(H)) (same W1 as eval_itransformer_pure_icl),
W2 ~ U(+-1/sqrt(R)) with a fixed seed.  The correction is g(q) - g0(q), so the
unadapted forecast is exactly the backbone.  On the demo loss
sum_j |v_j - (g - g0)(k_j)|^2 (v_j = backbone error) both layers take one step
from the same initial point (gradients at g = g0, residual = v_j):

  dW2 = eta2 sum_j v_j h_j^T
  dW1 = eta1 sum_j [(W2^T v_j) * 1(W1 k_j > 0)] k_j^T,   eta1 = a1 / sum_j |k_j|^2
  pred = f(x) + (W2 + dW2) ReLU((W1 + dW1) q) - W2 ReLU(W1 q)

eta2 = a2 / (|h_q| sum_j |h_j|) (cos) or a2 / sum_j |h_j|^2 (nlms).
Families per rule: W2 (a1 = 0), W1 (a2 = 0), joint (both > 0).  References on the
same demos: phase_mean, and the current pure ICL (zero W2, W2 then W1, W1 step
clipped at 0.1 |W1|; its own grids and families).  All chosen on val, test once.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_itransformer_tefl_icl_periodic import aligned_demo_ages
from patchmem.eval_itransformer_pure_icl import (
    A1_GRID as PURE_A1, A2_GRID as PURE_A2, FAMILIES as PURE_FAMILIES,
    forecast_errors_base, load_base, score_grid_pure, zero_output_head)
from patchmem.eval_itransformer_stepsize import CELLS

L, H, R = 96, 192, 64
A_GRID = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
JA1_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
CLIP = 0.1


def random_w2(seed, device):
    gen = torch.Generator(device='cpu').manual_seed(seed + 32000)
    W2 = torch.empty(H, R, dtype=torch.float64).uniform_(-1 / math.sqrt(R), 1 / math.sqrt(R), generator=gen)
    return W2.to(device)


def combos():
    out = [('mean', a, 0.0) for a in (0.0,) + A_GRID]
    for rule in ('cos', 'nlms'):
        out += [(rule, a2, a1) for a2 in (0.0,) + A_GRID for a1 in (0.0,) + JA1_GRID if a2 or a1]
    return out


@torch.inference_mode()
def score_chunk(q, K, V, target, W1, W2, cands):
    """Per-case MSE of the backbone + (g - g0)(q) after one joint step; target/V are backbone errors."""
    pre_q, pre_k = q @ W1.T, K @ W1.T                 # b C R / b C n R
    hq, hk = torch.relu(pre_q), torch.relu(pre_k)
    mask = (pre_k > 0).to(K.dtype)
    g0q = hq @ W2.T
    kq = (K * q[:, :, None]).sum(-1)                   # b C n
    ksq = K.square().sum((-1, -2)).clamp_min(1e-12)    # b C
    dq = (((V @ W2) * mask) * kq[..., None]).sum(-2) / ksq[..., None]   # dW1 q / a1
    den = {'cos': hq.norm(dim=-1) * hk.norm(dim=-1).sum(-1) + 1e-6,
           'nlms': hk.square().sum((-1, -2)).clamp_min(1e-12)}
    out = {}
    for a in sorted({c[1] for c in cands if c[0] == 'mean'}):
        out['mean', a, 0.0] = (target - a * V.mean(-2)).square().mean((1, 2)).cpu()
    for rule in ('cos', 'nlms'):
        for a1 in sorted({c[2] for c in cands if c[0] == rule}):
            hq1 = torch.relu(pre_q + a1 * dq)
            base = hq1 @ W2.T - g0q
            sim = (hk * hq1[:, :, None]).sum(-1)
            att = (sim[..., None] * V).sum(-2) / den[rule][..., None]
            for a2 in sorted({c[1] for c in cands if c[0] == rule and c[2] == a1}):
                out[rule, a2, a1] = (target - base - a2 * att).square().mean((1, 2)).cpu()
    return out


@torch.inference_mode()
def score_cell(E, origins, W1, W2, ages, chunk, cands):
    ages = torch.as_tensor(ages, dtype=torch.long)
    dev = W1.device
    res = {c: torch.empty(len(origins), dtype=torch.float64) for c in cands}
    for start in range(0, len(origins), chunk):
        t = origins[start:start + chunk]
        s = t[:, None] - ages[None]
        q = E[t - H - L].permute(0, 2, 1).to(device=dev, dtype=W1.dtype)
        target = E[t - L].permute(0, 2, 1).to(device=dev, dtype=W1.dtype)
        K = E[s - H - L].permute(0, 3, 1, 2).to(device=dev, dtype=W1.dtype)
        V = E[s - L].permute(0, 3, 1, 2).to(device=dev, dtype=W1.dtype)
        for c, m in score_chunk(q, K, V, target, W1, W2, cands).items():
            res[c][start:start + len(t)] = m
    return res


def families():
    fam = {'phase_mean': lambda c: c[0] == 'mean' and c[1] > 0}
    for rule in ('cos', 'nlms'):
        fam[f'{rule}:W2'] = lambda c, r=rule: c[0] == r and c[1] > 0 and c[2] == 0
        fam[f'{rule}:W1'] = lambda c, r=rule: c[0] == r and c[1] == 0 and c[2] > 0
        fam[f'{rule}:joint'] = lambda c, r=rule: c[0] == r and c[1] > 0 and c[2] > 0
    return fam


def run(code, seed, chunk, forecast_bs, output):
    import patchmem.eval_itransformer_pure_icl as pure_mod
    assert pure_mod.H == H, 'joint ICL is defined on the H=192 cells; unset ITR_H'
    exp, checkpoint, setting = load_base(code, seed)
    W1, W2zero = zero_output_head(seed, exp.device)
    W2 = random_w2(seed, exp.device)
    history, period = CELLS[code]
    ages = aligned_demo_ages(history, period)
    grid = combos()
    pure_grid = [(a1, a2) for a1 in PURE_A1 for a2 in PURE_A2]
    fams = families()
    chosen, pure_chosen, result, ref = {}, {}, {}, {}
    for split in ('val', 'test'):
        E, origins, channels = forecast_errors_base(exp, split, history, forecast_bs)
        picked = grid if split == 'val' else [x['combo'] for x in chosen.values()]
        cands = sorted(set(picked) | {('mean', 0.0, 0.0)})
        scores = score_cell(E, origins, W1, W2, ages, chunk, cands)
        pure_c = pure_grid if split == 'val' else sorted({x['combo'] for x in pure_chosen.values()})
        pure = score_grid_pure(E, origins, W1, W2zero, ages, chunk, pure_c, CLIP)
        base = float(scores['mean', 0.0, 0.0].mean())
        ref[split] = base
        print(f'[{split}] cases={len(origins)} channels={channels} demos={len(ages)} base={base:.6f}', flush=True)
        if split == 'val':
            for f, ok in fams.items():
                best = min((c for c in grid if ok(c)), key=lambda c: float(scores[c].mean()))
                chosen[f] = dict(combo=best, val_mse=float(scores[best].mean()))
            for f, ok in PURE_FAMILIES.items():
                best = min((c for c in pure_grid if ok(*c)), key=lambda c: float(pure[c].mean()))
                pure_chosen[f'pure:{f}'] = dict(combo=best, val_mse=float(pure[best].mean()))
        else:
            official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
            if abs(base - official) > 1e-4:
                raise AssertionError(f'base cached={base} official={official}')
            mean_case = scores[chosen['phase_mean']['combo']]
            allsc = {**{f: scores[x['combo']] for f, x in chosen.items()},
                     **{f: pure[x['combo']] for f, x in pure_chosen.items()}}
            for f, x in {**chosen, **pure_chosen}.items():
                m = allsc[f]
                result[f] = dict(combo=x['combo'], val_mse=x['val_mse'], test_mse=float(m.mean()),
                                 test_gain_pct=100 * (base - float(m.mean())) / base,
                                 finite=bool(torch.isfinite(m).all()),
                                 case_win_rate_vs_mean=float((m < mean_case).double().mean()))
                print(f'[test] {f:16s} {x["combo"]} mse={result[f]["test_mse"]:.6g} '
                      f'gain={result[f]["test_gain_pct"]:+.2f}% win_vs_mean={result[f]["case_win_rate_vs_mean"]:.3f}',
                      flush=True)
        del E
    out = dict(dataset=code, seed=seed, backbone='base', L=L, H=H, history=history, period=period,
               demo_ages=ages, w2_init='U(+-1/sqrt(R)), seed+32000', grids=dict(A=A_GRID, joint_a1=JA1_GRID,
               pure_a1=PURE_A1, pure_a2=PURE_A2, pure_clip=CLIP), checkpoint=str(checkpoint),
               val_ref=ref['val'], test_ref=ref['test'], families=result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, choices=CELLS)
    parser.add_argument('--seed', type=int, default=3047)
    parser.add_argument('--chunk', type=int, default=4)
    parser.add_argument('--forecast_bs', type=int, default=8)
    args = parser.parse_args()
    output = Path(f'patchmem/logs/server/itr_joint_icl/{args.data}_s{args.seed}.json')
    run(args.data, args.seed, args.chunk, args.forecast_bs, output)


if __name__ == '__main__':
    main()
