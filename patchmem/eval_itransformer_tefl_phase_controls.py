"""Periodicity vs in-context mapping: controls on identical periodic demos.

All arms correct the trained TEFL forecast g(q) with the TEFL residuals
r_j = E[s_j] - g(E[s_j-H]) of the same causal demos (ages = multiples of P).
They differ only in how the residuals are combined:

  icl_w2        W2-only in-context step (similarity-weighted, as in the method)
  icl_w2_shufK  same, similarity weights permuted among the demos (derangement);
                demos share the query phase, so the shuffle is phase-preserving
  phase_mean    a * mean_j r_j                        (no key dependence)
  phase_normw   a * sum_j |h_j| r_j / sum_j |h_j|     (icl_w2 with every cos = 1)
  phase_persist a * r_latest                          (most recent same-phase demo)
  ridge         a * kernel ridge on h = ReLU(W1 k), anchored at TEFL
  ridge_icpt    same with an unpenalized intercept (= phase mean + key part)

Every arm's hyperparameters are chosen on the official val split and the
chosen setting is evaluated once on test.

--backbone base runs the same arms on the standalone iTransformer (no TEFL):
g = 0 (zero W2, fixed random W1 as in eval_itransformer_pure_icl), so r_j is the
backbone error and icl_w2 equals the pure-ICL W2-only step.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from patchmem.eval_itransformer_tefl_icl import forecast_errors, load_model
from patchmem.eval_itransformer_tefl_icl_periodic import aligned_demo_ages
from patchmem.eval_itransformer_pure_icl import forecast_errors_base, load_base, zero_output_head

L = 96
H = int(os.environ.get('ITR_H', '192'))  # forecast horizon; set ITR_H for H != 192
A_GRID = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
RIDGE_LAM = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
RIDGE_A = (0.5, 1.0, 1.5, 2.0)


def derangement(shape, n, generator, device):
    """Random derangement of n demos per (case, channel): sigma o shift o sigma^-1."""
    sigma = torch.argsort(torch.rand(*shape, n, generator=generator), dim=-1)
    inv = torch.argsort(sigma, dim=-1)
    pi = torch.gather(sigma, -1, (inv + 1) % n)
    assert not (pi == torch.arange(n)).any()
    return pi.to(device)


def ridge_weights(G, kq, lam):
    n = G.shape[-1]
    tau = (G.diagonal(dim1=-2, dim2=-1).sum(-1) / n).clamp_min(1e-12)
    eye = torch.eye(n, dtype=G.dtype, device=G.device)
    A = G + lam * tau[..., None, None] * eye
    return torch.linalg.solve(A, kq[..., None]).squeeze(-1)


@torch.inference_mode()
def corrections(q, K, V, W1, W2, n_perm, generator):
    """Unit-scale corrections c (b, C, H) for every arm/setting, plus diagnostics."""
    hq = torch.relu(q @ W1.T)                       # b C R
    hk = torch.relu(K @ W1.T)                       # b C n R
    gq = hq @ W2.T                                  # b C H
    r = V - hk @ W2.T                               # b C n H
    n = K.shape[2]
    s = (hk * hq[:, :, None]).sum(-1)               # b C n
    hk_norm = hk.norm(dim=-1)
    den = hq.norm(dim=-1) * hk_norm.sum(-1) + 1e-6
    out = {'icl_w2': (s[..., None] * r).sum(-2) / den[..., None]}
    for p in range(n_perm):
        pi = derangement(s.shape[:2], n, generator, s.device)
        sp = torch.gather(s, -1, pi)
        out[f'icl_w2_shufK{p}'] = (sp[..., None] * r).sum(-2) / den[..., None]
    out['phase_mean'] = r.mean(-2)
    out['phase_normw'] = (hk_norm[..., None] * r).sum(-2) / (hk_norm.sum(-1)[..., None] + 1e-6)
    out['phase_persist'] = r[:, :, 0]               # ages ascending -> index 0 is latest
    G = hk @ hk.transpose(-1, -2)
    kq = (hk * hq[:, :, None]).sum(-1)
    hk_mean = hk.mean(-2, keepdim=True)
    hkc, hqc = hk - hk_mean, hq - hk_mean[:, :, 0]
    r_mean = r.mean(-2)
    rc = r - r_mean[:, :, None]
    Gc = hkc @ hkc.transpose(-1, -2)
    kqc = (hkc * hqc[:, :, None]).sum(-1)
    for lam in RIDGE_LAM:
        alpha = ridge_weights(G, kq, lam)
        out[f'ridge_lam{lam:g}'] = (alpha[..., None] * r).sum(-2)
        alpha_c = ridge_weights(Gc, kqc, lam)
        out[f'ridge_icpt_lam{lam:g}'] = r_mean + (alpha_c[..., None] * rc).sum(-2)
    # Diagnostics: how far the similarity weights are from uniform.
    cos = s / (hq.norm(dim=-1)[..., None] * hk_norm + 1e-12)
    w = s / (s.sum(-1, keepdim=True) + 1e-12)
    neff = 1.0 / (w.square().sum(-1) + 1e-12) / n
    live = (hq.norm(dim=-1) > 0) & (hk_norm.sum(-1) > 0)
    diag = {'cos_mean': cos.mean(-1), 'cos_std': cos.std(-1), 'neff_frac': neff, 'live': live.to(cos.dtype)}
    return gq, out, diag


def arm_of(name):
    for prefix in ('icl_w2_shufK', 'ridge_icpt', 'ridge'):
        if name.startswith(prefix):
            return prefix
    return name


def scales(name):
    return RIDGE_A if name.startswith('ridge') else A_GRID


@torch.inference_mode()
def collect(E, origins, W1, W2, ages, chunk, n_perm, seed):
    """Per-case sufficient stats so MSE(a) = S0 - 2a S1 + a^2 S2 for every correction."""
    ages = torch.as_tensor(ages, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed + 52000)
    dev = W1.device
    S0 = torch.empty(len(origins), dtype=torch.float64)
    S1, S2, D = {}, {}, {}
    for start in range(0, len(origins), chunk):
        t = origins[start:start + chunk]
        sl = slice(start, start + len(t))
        sidx = t[:, None] - ages[None]
        q = E[t - H - L].permute(0, 2, 1).to(device=dev, dtype=W1.dtype)
        target = E[t - L].permute(0, 2, 1).to(device=dev, dtype=W1.dtype)
        K = E[sidx - H - L].permute(0, 3, 1, 2).to(device=dev, dtype=W1.dtype)
        V = E[sidx - L].permute(0, 3, 1, 2).to(device=dev, dtype=W1.dtype)
        gq, cs, diag = corrections(q, K, V, W1, W2, n_perm, gen)
        tr = target - gq
        S0[sl] = tr.square().mean((1, 2)).cpu()
        for k, c in cs.items():
            if k not in S1:
                S1[k] = torch.empty(len(origins), dtype=torch.float64)
                S2[k] = torch.empty(len(origins), dtype=torch.float64)
            S1[k][sl] = (tr * c).mean((1, 2)).cpu()
            S2[k][sl] = c.square().mean((1, 2)).cpu()
        for k, v in diag.items():
            D.setdefault(k, torch.empty(len(origins), dtype=torch.float64))[sl] = v.mean(1).cpu()
    return S0, S1, S2, D


def settings_mse(S0, S1, S2):
    """Per-case MSE vectors for every (correction, scale)."""
    return {(k, a): S0 - 2 * a * S1[k] + a * a * S2[k] for k in S1 for a in scales(k)}


def run(code, seed, history, period, chunk, forecast_bs, n_perm, output, baseline_tag, backbone='tefl'):
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
    ages = aligned_demo_ages(history, period)
    assert ages == sorted(ages)
    per = {}
    for split in ('val', 'test'):
        E, origins, channels = errors(exp, split, history, forecast_bs)
        S0, S1, S2, D = collect(E, origins, W1, W2, ages, chunk, n_perm, seed)
        tefl = float(S0.mean())
        print(f'[{split}] cases={len(origins)} channels={channels} demos={len(ages)} TEFL={tefl:.6f}', flush=True)
        if split == 'test':
            official = json.loads(metrics.read_text())['mse']
            if abs(tefl - official) > 1e-4:
                raise AssertionError(f'{backbone} cached={tefl} official={official}')
        per[split] = dict(tefl=tefl, mse=settings_mse(S0, S1, S2),
                          diag={k: float(v.mean()) for k, v in D.items()})
        del E
    val, test = per['val'], per['test']
    # Select each arm's setting on val; shuffle perms share one selection rule
    # but each perm is selected and scored on its own.
    arms = {}
    for k in sorted({k for k, _ in val['mse']}):
        options = [(kk, a) for kk, a in val['mse'] if kk == k]
        best = min(options, key=lambda o: float(val['mse'][o].mean()))
        v, te = float(val['mse'][best].mean()), test['mse'][best]
        arms[k] = dict(scale=best[1], val_mse=v, test_mse=float(te.mean()),
                       test_gain_pct=100 * (test['tefl'] - float(te.mean())) / test['tefl'])
    groups = {}
    for k, v in arms.items():
        g = arm_of(k)
        if g in ('ridge', 'ridge_icpt'):
            if g not in groups or v['val_mse'] < groups[g]['val_mse']:
                groups[g] = {**v, 'setting': k}
        elif g == 'icl_w2_shufK':
            groups.setdefault(g, []).append(v)
        else:
            groups[g] = {**v, 'setting': k}
    shuf = groups['icl_w2_shufK']
    groups['icl_w2_shufK'] = dict(test_mse=float(np.mean([x['test_mse'] for x in shuf])),
                                  test_gain_pct=float(np.mean([x['test_gain_pct'] for x in shuf])),
                                  per_perm=shuf)
    # Paired per-case comparison of the method against each control on test.
    ref_key = (groups['icl_w2']['setting'], groups['icl_w2']['scale'])
    ref = test['mse'][ref_key]
    paired = {}
    for g in ('phase_mean', 'phase_normw', 'phase_persist', 'ridge', 'ridge_icpt'):
        other = test['mse'][(groups[g]['setting'], groups[g]['scale'])]
        d = (other - ref).numpy()
        paired[g] = dict(icl_win_rate=float((d > 0).mean()), mean_diff=float(d.mean()))
    for g, v in groups.items():
        print(f'[test] {g:14s} mse={v["test_mse"]:.6f} gain={v["test_gain_pct"]:+.2f}% '
              f'{v.get("setting", "")} a={v.get("scale", "")}', flush=True)
    print(f'[diag] val={val["diag"]} test={test["diag"]}', flush=True)
    result = dict(dataset=code, seed=seed, backbone=backbone,
                  model='iTransformer-TEFL' if backbone == 'tefl' else 'iTransformer (standalone)', L=L, H=H, L_error=H,
                  history=history, period=period, demo_ages=ages, n_perm=n_perm,
                  checkpoint=str(checkpoint), baseline_tag=baseline_tag,
                  val_tefl=val['tefl'], test_tefl=test['tefl'],
                  val_diag=val['diag'], test_diag=test['diag'],
                  summary=groups, paired_vs_icl_w2=paired, all_arms=arms)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='T')
    parser.add_argument('--seed', type=int, default=3047)
    parser.add_argument('--history', type=int, default=2880)
    parser.add_argument('--period', type=int, default=168)
    parser.add_argument('--chunk', type=int, default=4)
    parser.add_argument('--forecast_bs', type=int, default=8)
    parser.add_argument('--n_perm', type=int, default=3)
    parser.add_argument('--baseline_tag', default='TEFL_paper_L96_E192')
    parser.add_argument('--backbone', default='tefl', choices=('tefl', 'base'))
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    output = Path(args.out or f'patchmem/logs/server/itr_phase_controls/'
                              f'{args.data}_N{args.history}_P{args.period}_s{args.seed}'
                              f'{"_base" if args.backbone == "base" else ""}.json')
    run(args.data, args.seed, args.history, args.period, args.chunk, args.forecast_bs,
        args.n_perm, output, args.baseline_tag, args.backbone)


if __name__ == '__main__':
    main()
