"""Train a standalone iTransformer backbone for the no-TEFL ICL ablation."""
import argparse
import json
from pathlib import Path

import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from patchmem.diag_rollout import official_args
from patchmem.eval_itransformer_tefl_icl import CONFIG
from utils.tools import set_seed


@torch.inference_mode()
def streaming_test(exp, setting, code, seed):
    _, loader = exp._get_data(flag='test')
    exp.model.eval()
    squared = absolute = 0.0
    count = 0
    for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
        x = batch_x.float().to(exp.device)
        x_mark = batch_x_mark.float().to(exp.device)
        y = batch_y[:, -exp.args.pred_len:].float().to(exp.device)
        pred = exp.model(x, x_mark, None, None)
        delta = pred - y
        squared += delta.double().square().sum().item()
        absolute += delta.double().abs().sum().item()
        count += delta.numel()
    assert count > 0
    metrics = dict(dataset=CONFIG[code]['name'], model='iTransformer', seed=seed,
                   L=exp.args.seq_len, H=exp.args.pred_len, elements=count,
                   mse=squared / count, mae=absolute / count)
    out = Path(exp.args.results) / setting
    out.mkdir(parents=True, exist_ok=True)
    (out / 'base_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(f'[base test] {json.dumps(metrics)}', flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, choices=CONFIG)
    parser.add_argument('--seed', type=int, default=3047)
    parser.add_argument('--pred_len', type=int, default=192)
    parser.add_argument('--seq_len', type=int, default=96)
    args0 = parser.parse_args()
    code, seed, H, L = args0.data, args0.seed, args0.pred_len, args0.seq_len
    c = CONFIG[code]
    argv = (f'--task_name long_term_forecast --is_training 1 --model iTransformer '
            f'--model_id {c["name"]}_{L}_{H}_BASE --data {c["data"]} '
            f'--root_path {c["root"]} --data_path {c["file"]} --features M '
            f'--seq_len {L} --label_len 48 --pred_len {H} '
            f'--enc_in {c["channels"]} --dec_in {c["channels"]} --c_out {c["channels"]} '
            f'--e_layers {c["layers"]} --d_model {c["dim"]} --d_ff {c["ff"]} '
            f'--batch_size {c["batch"]} --learning_rate {c["lr"]} '
            f'--VPT_mode 0 --patience 3 --loss MSE --train_epochs 10 '
            f'--seed {seed} --itr 1 --des ICL_base_s{seed}').split()
    args = official_args(argv)
    set_seed(seed)
    setting = f'iTransformer_base_{code}_L{L}_H{H}_s{seed}'
    print(f'[base train] {setting} {json.dumps(vars(args), default=str)}', flush=True)
    exp = Exp_Long_Term_Forecast(args)
    exp.train(setting)
    streaming_test(exp, setting, code, seed)


if __name__ == '__main__':
    main()
