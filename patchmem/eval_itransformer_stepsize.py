"""Step-size rules for the case-wise W2 -> W1 GD step on the TEFL head.

Same demos (H=192 best periodic cell), same order (W2 step, then W1 step on the
post-W2 residual), only the step-size rule differs:

  cos   eta2 = a2 / (|h_q| sum_j |h_j|),  eta1 = a1 / sum_j |k_j|^2   (current method)
  nlms  eta2 = a2 / sum_j |h_j|^2,         eta1 = a1 / sum_j |k_j|^2   (demo-only normalisation)
  const eta2 = a2,                         eta1 = a1                   (plain GD, fixed lr)

  dW2 = eta2 sum_j v_j h_j^T,   v_j = r_j (TEFL residual; backbone error for --backbone base)
  dW1 = eta1 sum_j [((W2 + dW2)^T e_j) * 1(W1 k_j > 0)] k_j^T,   e_j = v_j - dW2 h_j
  pred = (W2 + dW2) ReLU((W1 + dW1) q)

phase_mean (a * mean_j r_j) is the reference.  For every rule two families,
W2 (a1 = 0) and W2+W1 (a1 > 0), are selected on the official val split; test
is scored once.  --backbone base uses the standalone iTransformer (fixed
random W1, zero W2) exactly as eval_itransformer_pure_icl.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_itransformer_tefl_icl import forecast_errors, load_model
from patchmem.eval_itransformer_tefl_icl_periodic import aligned_demo_ages
from patchmem.eval_itransformer_pure_icl import forecast_errors_base, load_base, zero_output_head

L, H = 96, 192
CELLS = {'E': (1440, 24), 'M1': (2880, 672), 'M2': (2880, 96), 'W': (2880, 144), 'T': (2880, 168)}
A_GRID = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
A1_GRID = (0.03, 0.1, 0.3, 1.0)
CONST_A2 = tuple(10.0 ** (k / 2) for k in range(-12, 1))      # 1e-6 .. 1, half decades
CONST_A1 = tuple(10.0 ** k for k in range(-6, 0))              # 1e-6 .. 1e-1
GRIDS = {'cos': (A_GRID, A1_GRID), 'nlms': (A_GRID, A1_GRID), 'const': (CONST_A2, CONST_A1)}


def combos():
    out = [('mean', a, 0.0) for a in A_GRID]
    for rule, (g2, g1) in GRIDS.items():
        out += [(rule, a2, a1) for a2 in g2 for a1 in (0.0,) + g1]
    return out


@torch.inference_mode()
def score_chunk(q, K, V, target, W1, W2, cands):
    """Per-case MSE for each (rule, a2, a1); dW2 kept implicit through Gram products."""
    pre_q, pre_k = q @ W1.T, K @ W1.T                # b C R / b C n R
    hq, hk = torch.relu(pre_q), torch.relu(pre_k)
    mask = (pre_k > 0).to(K.dtype)
    gq = hq @ W2.T
    r = V - hk @ W2.T                                 # b C n H
    G = hk @ hk.transpose(-1, -2)                     # b C n n
    kq = (K * q[:, :, None]).sum(-1)                  # b C n
    ksq = K.square().sum((-1, -2)).clamp_min(1e-12)   # b C
    den = {'cos': hq.norm(dim=-1) * hk.norm(dim=-1).sum(-1) + 1e-6,
           'nlms': G.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12),
           'const': torch.ones_like(ksq)}
    w1_den = {'cos': ksq, 'nlms': ksq, 'const': torch.ones_like(ksq)}
    out = {}
    for a in sorted({c[1] for c in cands if c[0] == 'mean'}):
        out['mean', a, 0.0] = (target - gq - a * r.mean(-2)).square().mean((1, 2)).cpu()
    for rule in ('cos', 'nlms', 'const'):
        for a2 in sorted({c[1] for c in cands if c[0] == rule}):
            eta = (a2 / den[rule])[..., None, None]
            e = r - eta * (G @ r)                     # dW2 h_j = eta sum_i <h_i, h_j> r_i
            eW2 = e @ W2 + eta * ((e @ r.transpose(-1, -2)) @ hk)
            dq = ((eW2 * mask) * kq[..., None]).sum(-2) / w1_den[rule][..., None]
            for a1 in sorted({c[2] for c in cands if c[:2] == (rule, a2)}):
                hq1 = torch.relu(pre_q + a1 * dq)
                sim = (hk * hq1[:, :, None]).sum(-1)
                pred = hq1 @ W2.T + eta[..., 0] * (sim[..., None] * r).sum(-2)
                out[rule, a2, a1] = (target - pred).square().mean((1, 2)).cpu()
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
    fam = {'phase_mean': lambda c: c[0] == 'mean'}
    for rule in GRIDS:
        fam[f'{rule}:W2'] = lambda c, r=rule: c[0] == r and c[2] == 0.0
        fam[f'{rule}:W2+W1'] = lambda c, r=rule: c[0] == r and c[2] > 0.0
    return fam


def run(code, seed, backbone, chunk, forecast_bs, output, baseline_tag):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if backbone == 'tefl':
        exp, checkpoint = load_model(code, seed, device, baseline_tag)
        W1 = exp.model.W1.weight.detach().to(device=device, dtype=torch.float64)
        W2 = exp.model.W2.weight.detach().to(device=device, dtype=torch.float64)
        errors = forecast_errors
        metrics = checkpoint.parents[1].parent / 'results' / checkpoint.parent.name / 'tefl_metrics.json'
    else:
        exp, checkpoint, setting = load_base(code, seed)
        W1, W2 = zero_output_head(seed, exp.device)
        errors = forecast_errors_base
        metrics = Path(exp.args.results) / setting / 'base_metrics.json'
    history, period = CELLS[code]
    ages = aligned_demo_ages(history, period)
    grid = combos()
    fams = families()
    chosen, result, ref = {}, {}, {}
    for split in ('val', 'test'):
        E, origins, channels = errors(exp, split, history, forecast_bs)
        picked = grid if split == 'val' else [x['combo'] for x in chosen.values()]
        cands = sorted(set(picked) | {('mean', 0.0, 0.0)})
        scores = score_cell(E, origins, W1, W2, ages, chunk, cands)
        base = float(scores['mean', 0.0, 0.0].mean())
        ref[split] = base
        print(f'[{split}] cases={len(origins)} channels={channels} demos={len(ages)} {backbone}={base:.6f}', flush=True)
        if split == 'val':
            for f, ok in fams.items():
                best = min((c for c in grid if ok(c)), key=lambda c: float(scores[c].mean()))
                chosen[f] = dict(combo=best, val_mse=float(scores[best].mean()))
        else:
            official = json.loads(metrics.read_text())['mse']
            if abs(base - official) > 1e-4:
                raise AssertionError(f'{backbone} cached={base} official={official}')
            mean_case = scores[chosen['phase_mean']['combo']]
            for f, x in chosen.items():
                m = scores[x['combo']]
                result[f] = dict(rule=x['combo'][0], a2=x['combo'][1], a1=x['combo'][2], val_mse=x['val_mse'],
                                 test_mse=float(m.mean()), test_gain_pct=100 * (base - float(m.mean())) / base,
                                 finite=bool(torch.isfinite(m).all()),
                                 case_win_rate_vs_mean=float((m < mean_case).double().mean()))
                print(f'[test] {f:12s} {x["combo"]} mse={result[f]["test_mse"]:.6g} '
                      f'gain={result[f]["test_gain_pct"]:+.2f}% win_vs_mean={result[f]["case_win_rate_vs_mean"]:.3f}',
                      flush=True)
        del E
    out = dict(dataset=code, seed=seed, backbone=backbone, L=L, H=H, history=history, period=period,
               demo_ages=ages, grids=dict(A=A_GRID, A1=A1_GRID, const_a2=CONST_A2, const_a1=CONST_A1),
               checkpoint=str(checkpoint), val_ref=ref['val'], test_ref=ref['test'], families=result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, choices=CELLS)
    parser.add_argument('--seed', type=int, default=3047)
    parser.add_argument('--backbone', default='tefl', choices=('tefl', 'base'))
    parser.add_argument('--chunk', type=int, default=4)
    parser.add_argument('--forecast_bs', type=int, default=8)
    parser.add_argument('--baseline_tag', default='TEFL_paper_L96_E192')
    args = parser.parse_args()
    output = Path(f'patchmem/logs/server/itr_stepsize/{args.data}_s{args.seed}_{args.backbone}.json')
    run(args.data, args.seed, args.backbone, args.chunk, args.forecast_bs, output, args.baseline_tag)


if __name__ == '__main__':
    main()
