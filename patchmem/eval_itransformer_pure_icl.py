"""Two-layer case-wise ICL on a standalone iTransformer, without TEFL training.

W1 is a fixed random feature map and W2 starts at zero, so the unadapted
forecast is exactly the backbone.  For each case, update W2 from the observed
demonstrations, then update W1 using that newly nonzero W2.  Both updates use
only causally available forecast errors.  Validation selects step sizes.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from patchmem.diag_rollout import official_args
from patchmem.eval_itransformer_tefl_icl import CONFIG
from patchmem.eval_itransformer_tefl_icl_periodic import aligned_demo_ages


ROOT = Path('patchmem/logs/server/itr_pure_icl')
L, RANK = 96, 64
H = int(os.environ.get('ITR_H', '192'))  # forecast horizon; set ITR_H for H != 192
A1_GRID = (0.0, 0.001, 0.01, 0.03, 0.1, 1.0, 10.0)
A2_GRID = (0.1, 0.3, 1.0, 3.0)
FAMILIES = {'W1+W2': lambda a1,a2: a1 > 0 and a2 > 0,
            'W2_only': lambda a1,a2: a1 == 0 and a2 > 0}


def load_base(code, seed):
    c = CONFIG[code]
    argv = (f'--task_name long_term_forecast --is_training 0 --model iTransformer '
            f'--model_id {c["name"]}_96_{H}_BASE --data {c["data"]} '
            f'--root_path {c["root"]} --data_path {c["file"]} --features M '
            f'--seq_len {L} --label_len 48 --pred_len {H} '
            f'--enc_in {c["channels"]} --dec_in {c["channels"]} --c_out {c["channels"]} '
            f'--e_layers {c["layers"]} --d_model {c["dim"]} --d_ff {c["ff"]} '
            f'--batch_size {c["batch"]} --learning_rate {c["lr"]} '
            f'--VPT_mode 0 --seed {seed} --des ICL_base_s{seed}').split()
    args = official_args(argv)
    exp = Exp_Long_Term_Forecast(args)
    setting = f'iTransformer_base_{code}_L96_H{H}_s{seed}'
    checkpoint = Path(args.checkpoints) / setting / 'checkpoint.pth'
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    exp.model.load_state_dict(torch.load(checkpoint, map_location=exp.device, weights_only=True))
    exp.model.eval()
    return exp, checkpoint, setting


@torch.inference_mode()
def forecast_errors_base(exp, split, max_history, batch_size):
    """Cache standalone-backbone errors for each fully observed causal origin."""
    args = exp.args
    old_seq_len = args.seq_len
    args.seq_len = max_history
    try:
        ds, _ = data_provider(args, split)
    finally:
        args.seq_len = old_seq_len
    X = torch.as_tensor(np.asarray(ds.data_x), dtype=torch.float32)
    marks = torch.as_tensor(np.asarray(ds.data_stamp), dtype=torch.float32)
    origins = torch.arange(L, len(X) - H + 1)
    E = torch.empty((len(origins), H, X.shape[1]), dtype=torch.float32)
    offsets = torch.arange(-L, 0)
    leads = torch.arange(H)
    for start in range(0, len(origins), batch_size):
        o = origins[start:start + batch_size]
        pred = exp.model(X[o[:, None] + offsets].to(exp.device),
                         marks[o[:, None] + offsets].to(exp.device), None, None)
        E[start:start + len(o)] = X[o[:, None] + leads] - pred.cpu()
    official_origins = torch.arange(max_history, max_history + len(ds))
    assert official_origins[-1] + H == len(X)
    return E, official_origins, X.shape[1]


def zero_output_head(seed, device, rank=RANK):
    generator = torch.Generator(device='cpu').manual_seed(seed + 31000)
    W1 = torch.empty(rank, H, dtype=torch.float64)
    W2 = torch.zeros(H, rank, dtype=torch.float64)
    # W1 cannot be zero: ReLU(W1 x) would otherwise give W2 zero gradient.
    W1.uniform_(-1 / math.sqrt(H), 1 / math.sqrt(H), generator=generator)
    return W1.to(device), W2.to(device)


def cell_ages(history, rule):
    if rule.startswith('D'):
        stride = int(rule[1:])
        return list(range(H, history - H - L + 1, stride))
    if rule.startswith('P'):
        return aligned_demo_ages(history, int(rule[1:]))
    raise ValueError(rule)


@torch.inference_mode()
def score_grid_pure(E, origins, W1, W2, ages, chunk, candidates, clip_w1=None):
    """Per-case MSE after W2 then W1 steps from a zero-output head."""
    ages = torch.as_tensor(ages, dtype=torch.long)
    assert len(ages) > 0 and int(ages.min()) >= H
    assert torch.count_nonzero(W2) == 0
    out = {pair: torch.empty(len(origins), dtype=torch.float64) for pair in candidates}
    device = W1.device
    for start in range(0, len(origins), chunk):
        t = origins[start:start + chunk]
        s = t[:, None] - ages[None]
        q = E[t - H - L].permute(0, 2, 1).to(device=device, dtype=W1.dtype)
        target = E[t - L].permute(0, 2, 1).to(device=device, dtype=W1.dtype)
        K = E[s - H - L].permute(0, 3, 1, 2).to(device=device, dtype=W1.dtype)
        V = E[s - L].permute(0, 3, 1, 2).to(device=device, dtype=W1.dtype)
        pre_k = K @ W1.T
        pre_q = q @ W1.T
        h0_k, h0_q = torch.relu(pre_k), torch.relu(pre_q)
        den2 = h0_q.norm(dim=-1) * h0_k.norm(dim=-1).sum(-1) + 1e-6
        W2_core = torch.einsum('bcnh,bcnr->bchr', V, h0_k) / den2[..., None, None]
        norm1 = K.square().sum((-1, -2)).clamp_min(1e-12)
        query_sim = (q[:, :, None, :] @ K.transpose(-1, -2)).squeeze(-2)
        for a2 in sorted({pair[1] for pair in candidates}):
            W2_new = a2 * W2_core
            residual = V - torch.einsum('bcnr,bchr->bcnh', h0_k, W2_new)
            neg_grad = torch.einsum('bcnh,bchr->bcnr', residual, W2_new)
            neg_grad *= (pre_k > 0).to(K.dtype)
            if clip_w1 is None:
                delta_q = torch.einsum('bcn,bcnr->bcr', query_sim, neg_grad) / norm1[..., None]
            else:
                delta_W1 = torch.einsum('bcnr,bcnh->bcrh', neg_grad, K) / norm1[..., None, None]
            for a1 in sorted({pair[0] for pair in candidates if pair[1] == a2}):
                if clip_w1 is None:
                    step_q = a1 * delta_q
                else:
                    step_W1 = a1 * delta_W1
                    step_norm = step_W1.square().sum((-1, -2)).sqrt().clamp_min(1e-12)
                    factor = ((clip_w1 * W1.norm()) / step_norm).clamp(max=1)
                    step_W1 = step_W1 * factor[..., None, None]
                    step_q = torch.einsum('bch,bcrh->bcr', q, step_W1)
                hq_new = torch.relu(pre_q + step_q)
                pred = torch.einsum('bcr,bchr->bch', hq_new, W2_new)
                out[a1, a2][start:start + len(t)] = (target - pred).square().mean((1, 2)).cpu()
    return out


def base_mse(E, origins):
    return float(E[origins - L].double().square().mean())


def run(code, seed, cells, chunk, forecast_bs, output, clip_w1=None):
    exp, checkpoint, setting = load_base(code, seed)
    W1, W2 = zero_output_head(seed, exp.device)
    candidates = [(a1, a2) for a1 in A1_GRID for a2 in A2_GRID]
    specs = [(n, rule, cell_ages(n, rule)) for n, rule in cells]
    assert all(ages and max(ages) <= n - H - L for n, _, ages in specs)
    validation, test, counts = {}, {}, {}
    for split in ('val', 'test'):
        E, origins, channels = forecast_errors_base(exp, split, max(n for n,_,_ in specs), forecast_bs)
        counts[split] = len(origins)
        baseline = base_mse(E, origins)
        print(f'[{split}] cases={len(origins)} channels={channels} base_MSE={baseline:.8f}', flush=True)
        if split == 'test':
            official = json.loads((Path(exp.args.results) / setting / 'base_metrics.json').read_text())['mse']
            if abs(baseline - official) > 1e-4:
                raise AssertionError(f'Cached base MSE {baseline} != official {official}')
        for n, rule, ages in specs:
            key = f'N{n}_{rule}'
            if split == 'val':
                scores = score_grid_pure(E, origins, W1, W2, ages, chunk, candidates, clip_w1)
                chosen = {}
                for family, admissible in FAMILIES.items():
                    options = [pair for pair in candidates if admissible(*pair)]
                    best = min(options, key=lambda pair: float(scores[pair].mean()))
                    mse = float(scores[best].mean())
                    chosen[family] = dict(a1=best[0],a2=best[1],mse=mse,
                                          val_prefers_update=mse < baseline)
                validation[key] = dict(history=n,rule=rule,n_demos=len(ages),
                                       base_mse=baseline,chosen=chosen)
                print(f'[val] {key} demos={len(ages)} joint={chosen["W1+W2"]}', flush=True)
            else:
                picked = validation[key]['chosen']
                selected = [(v['a1'],v['a2']) for v in picked.values()]
                scores = score_grid_pure(E, origins, W1, W2, ages, chunk, selected, clip_w1)
                arms = {}
                for family, choice in picked.items():
                    mse = float(scores[choice['a1'],choice['a2']].mean())
                    arms[family] = {**choice, 'mse':mse,
                                    'gain_pct':100 * (baseline-mse)/baseline,
                                    'guarded_gain_pct':100 * (baseline-mse)/baseline if choice['val_prefers_update'] else 0.0}
                test[key] = dict(history=n,rule=rule,n_demos=len(ages),
                                 base_mse=baseline,arms=arms)
                print(f'[test] {key} base={baseline:.6f} joint={arms["W1+W2"]["mse"]:.6f} '
                      f'gain={arms["W1+W2"]["gain_pct"]:+.2f}%', flush=True)
        del E
    result = dict(dataset=code,seed=seed,model='standalone iTransformer + two-layer ICL',
                  L=L,H=H,L_error=H,head_rank=W1.shape[0],
                  initialization='fixed random W1; zero W2; case-wise W2 then W1 update',
                  a1_grid=A1_GRID,a2_grid=A2_GRID,clip_w1=clip_w1,
                  checkpoint=str(checkpoint),validation_windows=counts['val'],
                  test_windows=counts['test'],validation=validation,test=test)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2) + '\n')
    print(f'[done] {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',required=True,choices=CONFIG)
    parser.add_argument('--seed',type=int,default=3047)
    parser.add_argument('--cells',required=True,help='e.g. 1440:D48,2880:P144')
    parser.add_argument('--chunk',type=int,default=8)
    parser.add_argument('--forecast_bs',type=int,default=16)
    parser.add_argument('--clip_w1',type=float,default=None,
                        help='maximum Frobenius norm of case W1 step as a fraction of initial W1 norm')
    parser.add_argument('--out',default=None)
    args = parser.parse_args()
    cells = []
    for spec in args.cells.split(','):
        history,rule = spec.split(':')
        cells.append((int(history),rule))
    output = Path(args.out or ROOT / f'{args.data}_pure_icl_s{args.seed}.json')
    run(args.data,args.seed,cells,args.chunk,args.forecast_bs,output,args.clip_w1)


if __name__ == '__main__':
    main()
