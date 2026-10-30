"""Evaluate per-case W1/W2 updates on trained iTransformer-TEFL checkpoints.

Each case uses its own N-step observed history.  A demonstration at origin s
has key E[s-H], target E[s], and value R[s] = E[s] - g(E[s-H]), where E is the
f-only forecast error and g is trained TEFL.  The in-context update uses only
demonstrations whose target has matured by the query origin.  Step sizes are
selected on the official validation split and evaluated on the official test
split.  No test targets are used for fitting or choosing hyperparameters.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
np.Inf = np.inf
from data_provider.data_factory import data_provider
from patchmem.diag_rollout import official_args
from patchmem.exp_itransformer_tefl import Exp_iTransformerTefl


CONFIG = {
    'E': dict(name='ETTh2', data='ETTh2', root='./dataset/ETT-small', file='ETTh2.csv',
              channels=7, layers=2, dim=128, ff=128, batch=32, lr=0.0001),
    'M1': dict(name='ETTm1', data='ETTm1', root='./dataset/ETT-small', file='ETTm1.csv',
               channels=7, layers=2, dim=128, ff=128, batch=32, lr=0.0001),
    'M2': dict(name='ETTm2', data='ETTm2', root='./dataset/ETT-small', file='ETTm2.csv',
               channels=7, layers=2, dim=128, ff=128, batch=32, lr=0.0001),
    'W': dict(name='weather', data='custom', root='./dataset/weather', file='weather.csv',
              channels=21, layers=3, dim=512, ff=512, batch=32, lr=0.0001),
    'T': dict(name='traffic', data='custom', root='./dataset/traffic', file='traffic.csv',
              channels=862, layers=4, dim=512, ff=512, batch=16, lr=0.001),
    'EL': dict(name='electricity', data='custom', root='./dataset/electricity', file='electricity.csv',
               channels=321, layers=3, dim=512, ff=512, batch=16, lr=0.0005),      # official iTransformer ECL recipe
}
A1_GRID = (0.0, 0.03, 0.1, 0.3, 1.0)
A2_GRID = (0.0, 0.1, 0.3, 1.0, 3.0)
FAMILIES = {
    'W1+W2': lambda a1, a2: a1 > 0 and a2 > 0,
    'W2_only': lambda a1, a2: a1 == 0 and a2 > 0,
    'W1_only': lambda a1, a2: a1 > 0 and a2 == 0,
}


def make_args(code, seed, baseline_tag):
    c = CONFIG[code]
    argv = (f'--task_name long_term_forecast --is_training 0 --model iTransformer '
            f'--model_id {c["name"]}_96_192_TEFL --data {c["data"]} '
            f'--root_path {c["root"]} --data_path {c["file"]} --features M '
            f'--seq_len 96 --label_len 48 --pred_len 192 '
            f'--enc_in {c["channels"]} --dec_in {c["channels"]} --c_out {c["channels"]} '
            f'--e_layers {c["layers"]} --d_model {c["dim"]} --d_ff {c["ff"]} '
            f'--batch_size {c["batch"]} --learning_rate {c["lr"]} '
            f'--VPT_mode 0 --patience 3 --seed {seed} --itr 1 --des {baseline_tag}_s{seed}').split()
    return official_args(argv)


def load_model(code, seed, device, baseline_tag):
    args = make_args(code, seed, baseline_tag)
    exp = Exp_iTransformerTefl(args)
    pattern = (f'checkpoints/long_term_forecast_{CONFIG[code]["name"]}_96_192_TEFL_'
               f'iTransformer_*_sl288_*_{baseline_tag}_s{seed}_0/checkpoint.pth')
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise FileNotFoundError(f'Expected one checkpoint, found {len(hits)}: {pattern}')
    checkpoint = Path(hits[0])
    exp.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    exp.model.eval()
    return exp, checkpoint


@torch.inference_mode()
def forecast_errors(exp, split, max_history, batch_size):
    """Cache f-only errors for every fully observed origin in the official split's prefix."""
    args = exp.args
    L, H = exp.model.L, exp.model.H
    args.seq_len = max_history
    try:
        ds, _ = data_provider(args, split)
    finally:
        args.seq_len = L + H
    X = torch.as_tensor(np.asarray(ds.data_x), dtype=torch.float32)
    marks = torch.as_tensor(np.asarray(ds.data_stamp), dtype=torch.float32)
    assert X.shape[0] == marks.shape[0]
    origins = torch.arange(L, len(X) - H + 1)
    E = torch.empty((len(origins), H, X.shape[1]), dtype=torch.float32)
    offsets = torch.arange(-L, 0)
    for start in range(0, len(origins), batch_size):
        o = origins[start:start + batch_size]
        idx = o[:, None] + offsets[None]
        pred = exp.model.base(X[idx].to(exp.device), marks[idx].to(exp.device), None, None)
        E[start:start + len(o)] = X[o[:, None] + torch.arange(H)] - pred.cpu()
    official_origins = torch.arange(max_history, max_history + len(ds))
    assert official_origins[-1] + H == len(X)
    return E, official_origins, X.shape[1]


def step_components(q, K, V, W1, W2):
    """Backpropagate output residuals into W1's low-rank hidden space.

    V and residual have horizon dimension H; neg_grad has hidden rank R.
    This is a loss-gradient step, not a regression on residuals as latent values.
    """
    pre_k = K @ W1.T
    g_old = torch.relu(pre_k) @ W2.T
    residual = V - g_old  # actual TEFL forecast residual, not the f-only error
    neg_grad = (residual @ W2) * (pre_k > 0).to(K.dtype)
    norm = K.square().sum((-1, -2)).clamp_min(1e-12)[..., None, None]
    # Associate the products to avoid the quadratic demo-by-demo matrix.
    delta_k = K @ (K.transpose(-1, -2) @ neg_grad) / norm
    delta_q = ((q[:, :, None, :] @ K.transpose(-1, -2)) @ neg_grad).squeeze(2) / norm[..., 0]
    return pre_k, residual, g_old, delta_k, delta_q


def updated_correction(q, K, V, W1, W2, a1, a2):
    """Implicit case-specific W1/W2 steps; V is f-only error, residual is explicit."""
    pre_k, residual, g_old, delta_k, delta_q = step_components(q, K, V, W1, W2)
    hk = torch.relu(pre_k + a1 * delta_k)
    hq = torch.relu(q @ W1.T + a1 * delta_q)
    out = hq @ W2.T
    if a2:
        # Recompute TEFL's residual after the W1 step.  Starting from the
        # already corrected value avoids subtracting the old g a second time.
        residual_after_w1 = residual + g_old - hk @ W2.T
        scores = (hk * hq[:, :, None]).sum(-1)
        den = hq.norm(dim=-1) * hk.norm(dim=-1).sum(-1) + 1e-6
        out = out + a2 * (scores[..., None] * residual_after_w1).sum(-2) / den[..., None]
    return out


@torch.inference_mode()
def score_grid(E, origins, W1, W2, N, stride, chunk, candidates, demo_ages=None):
    """Per-window MSE for fixed-stride or causally selected demo ages."""
    L, H = 96, 192
    if demo_ages is None:
        n = (N - L - 2 * H) // stride + 1
        ages = H + stride * torch.arange(n)
    else:
        ages = torch.as_tensor(demo_ages, dtype=torch.long)
        n = len(ages)
        assert n and int(ages.min()) >= H and int(ages.max()) <= N - H - L
    assert n >= 1
    out = {p: torch.empty(len(origins), dtype=torch.float64) for p in candidates}
    device = W1.device
    # CPU resident error cache; only the current case minibatch goes to GPU.
    for start in range(0, len(origins), chunk):
        t = origins[start:start + chunk]
        s = t[:, None] - ages[None]
        assert int((s - H - L).min()) >= int((t - N).min())
        assert int((s + H).max()) <= int(t.max())
        q = E[t - H - L].permute(0, 2, 1).to(device=device, dtype=W1.dtype)
        target = E[t - L].permute(0, 2, 1).to(device=device, dtype=W1.dtype)
        K = E[s - H - L].permute(0, 3, 1, 2).to(device=device, dtype=W1.dtype)
        V = E[s - L].permute(0, 3, 1, 2).to(device=device, dtype=W1.dtype)
        pre_k, residual, g_old, delta_k, delta_q = step_components(q, K, V, W1, W2)
        pre_q = q @ W1.T
        for a1 in sorted({p[0] for p in candidates}):
            hk = torch.relu(pre_k + a1 * delta_k)
            hq = torch.relu(pre_q + a1 * delta_q)
            base = hq @ W2.T
            residual_after_w1 = residual + g_old - hk @ W2.T
            scores = (hk * hq[:, :, None]).sum(-1)
            den = hq.norm(dim=-1) * hk.norm(dim=-1).sum(-1) + 1e-6
            add = (scores[..., None] * residual_after_w1).sum(-2) / den[..., None]
            for a2 in sorted({p[1] for p in candidates if p[0] == a1}):
                pred = base + a2 * add
                out[a1, a2][start:start + len(t)] = (target - pred).square().mean((1, 2)).cpu()
    return out, n


@torch.inference_mode()
def score_tefl(E, origins, W1, W2, chunk):
    out = torch.empty(len(origins), dtype=torch.float64)
    for start in range(0, len(origins), chunk):
        t = origins[start:start + chunk]
        q = E[t - 192 - 96].permute(0, 2, 1).to(device=W1.device, dtype=W1.dtype)
        target = E[t - 96].permute(0, 2, 1).to(device=W1.device, dtype=W1.dtype)
        pred = torch.relu(q @ W1.T) @ W2.T
        out[start:start + len(t)] = (target - pred).square().mean((1, 2)).cpu()
    return float(out.mean())


def run(code, seed, histories, strides, chunk, forecast_bs, output, baseline_tag='TEFL_L96_E192'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, checkpoint = load_model(code, seed, device, baseline_tag)
    W1 = exp.model.W1.weight.detach().to(device=device, dtype=torch.float64)
    W2 = exp.model.W2.weight.detach().to(device=device, dtype=torch.float64)
    print(f'[model] {checkpoint} device={device} histories={histories} strides={strides}', flush=True)
    candidates = [(a1, a2) for a1 in A1_GRID for a2 in A2_GRID if a1 or a2]
    val_results = {}
    test_results = {}
    split_counts = {}
    for split in ('val', 'test'):
        E, origins, channels = forecast_errors(exp, split, max(histories), forecast_bs)
        split_counts[split] = len(origins)
        print(f'[{split}] cases={len(origins)} channels={channels} error_cache_GB={E.numel() * 4 / 1e9:.2f}', flush=True)
        baseline = score_tefl(E, origins, W1, W2, chunk)
        if split == 'test':
            metrics = checkpoint.parents[1].parent / 'results' / checkpoint.parent.name / 'tefl_metrics.json'
            if metrics.exists():
                official = json.loads(metrics.read_text())['mse']
                print(f'[sanity] official={official:.8f} cached={baseline:.8f}', flush=True)
                if abs(baseline - official) > 1e-4:
                    raise AssertionError(f'TEFL baseline mismatch: {baseline} vs {official}')
        for N in histories:
            for stride in strides:
                key = f'N{N}_D{stride}'
                if split == 'val':
                    scores, n = score_grid(E, origins, W1, W2, N, stride, chunk, candidates)
                    chosen = {}
                    for fam, admissible in FAMILIES.items():
                        valid = [p for p in candidates if admissible(*p)]
                        best = min(valid, key=lambda p: float(scores[p].mean()))
                        chosen[fam] = {'a1': best[0], 'a2': best[1], 'mse': float(scores[best].mean())}
                    val_results[key] = {'N': N, 'stride': stride, 'n_demos': n, 'tefl_mse': baseline,
                                        'chosen': chosen}
                    print(f'[val] {key} demos={n} W1+W2={chosen["W1+W2"]}', flush=True)
                else:
                    selected = [(item['a1'], item['a2']) for item in val_results[key]['chosen'].values()]
                    scores, n = score_grid(E, origins, W1, W2, N, stride, chunk, selected)
                    arms = {}
                    for fam, item in val_results[key]['chosen'].items():
                        mse = float(scores[item['a1'], item['a2']].mean())
                        arms[fam] = {**item, 'mse': mse, 'gain_pct': 100 * (baseline - mse) / baseline}
                    test_results[key] = {'N': N, 'stride': stride, 'n_demos': n,
                                         'tefl_mse': baseline, 'arms': arms}
                    print(f'[test] {key} demos={n} TEFL={baseline:.6f} '
                          f'W1+W2={arms["W1+W2"]["mse"]:.6f} '
                          f'gain={arms["W1+W2"]["gain_pct"]:+.2f}%', flush=True)
        del E
    result = {'dataset': code, 'name': CONFIG[code]['name'], 'model': 'iTransformer-TEFL',
              'seed': seed, 'L': 96, 'H': 192, 'L_error': 192,
              'baseline_tag': baseline_tag,
              'checkpoint': str(checkpoint), 'validation_windows': split_counts['val'],
              'test_windows': split_counts['test'],
              'validation': val_results, 'test': test_results,
              'step_selection': 'per N/stride/family on official validation; test untouched'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, choices=CONFIG)
    parser.add_argument('--seed', type=int, default=3047)
    parser.add_argument('--histories', default='1440,2880')
    parser.add_argument('--strides', default='48,96,144,192,288,384')
    parser.add_argument('--chunk', type=int, default=16)
    parser.add_argument('--forecast_bs', type=int, default=16)
    parser.add_argument('--baseline_tag', default='TEFL_L96_E192')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    histories = [int(v) for v in args.histories.split(',')]
    strides = [int(v) for v in args.strides.split(',')]
    assert histories and strides and all(n >= 96 + 2 * 192 for n in histories)
    assert args.chunk > 0 and args.forecast_bs > 0
    out = Path(args.out or f'patchmem/logs/server/itr_tefl_icl/{args.data}_{args.baseline_tag}_s{args.seed}.json')
    run(args.data, args.seed, histories, strides, args.chunk, args.forecast_bs, out, args.baseline_tag)


if __name__ == '__main__':
    main()
