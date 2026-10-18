"""PAT-304: lookback / horizon generalisation outside the training range. Eval-only on a trained R1 / R1Mem checkpoint (no retraining):
the model is trained with L in [384, 1440], H in [96, 336]; here it is tested at lookbacks L outside that range (shorter and longer) and at
horizons beyond it (more rollout steps than ever trained). Same eval protocol as Exp_R1.test (eval_grid, test split built with seq_len=L,
pred_len=H, the checkpoint's fixed ATSP order). Window sets differ with (L, H), so compare ARMS at the same (L, H), not cells with each other.
usage (repo root): python patchmem/ood_eval.py --arm A1|Q8 --seed 3047 [--L 192,288,720,1920,2880] [--H 96,336,720] [--stride 1]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
np.Inf = np.inf
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from patchmem.diag_rollout import official_args

CK = 'long_term_forecast_weather_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue_{}_0'
ARMS = {'A1': ('R1Wh0I1000s{}', {}),
        'Q8': ('R1WMK0.5Q8c1h0I1000s{}', {'R1_MEM': '1', 'R1_MEM_QK_DIM': '8', 'R1_MEM_SLOT_KEEP': '0.5'})}
WEATHER = ('--task_name long_term_forecast --is_training 0 --root_path ./dataset/weather --data_path weather.csv --enc_in 21 --dec_in 21 '
           '--c_out 21 --model_id weather_r1 --model MambaTS --data custom --features M --seq_len 1440 --label_len 48 --pred_len 336 '
           '--e_layers 3 --d_layers 2 --factor 1 --itr 1 --n_heads 16 --d_model 32 --d_ff 512 --dropout 0.2 --patch_len 48 --stride 48 '
           '--VPT_mode 1 --ATSP_solver SA --batch_size 8 --learning_rate 0.0005').split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(ARMS)); ap.add_argument('--seed', type=int, default=3047)
    ap.add_argument('--L', default='192,288,720,1920,2880'); ap.add_argument('--H', default='96,336,720')
    ap.add_argument('--stride', type=int, default=1); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    des, env = ARMS[a.arm]; des = des.format(a.seed)
    os.environ.update(env)
    from patchmem.exp_r1 import Exp_R1
    args = official_args(WEATHER + ['--seed', str(a.seed), '--des', des])
    exp = Exp_R1(args)
    m = exp.model
    m.load_state_dict(torch.load(os.path.join(args.checkpoints, CK.format(des), 'checkpoint.pth'), map_location=exp.device))
    m.eval(); m.reset_ids_shuffle(); exp._fix_eval_order()
    grid = [(int(L), int(H)) for L in a.L.split(',') for H in a.H.split(',')]
    res = {}
    for g in grid:   # one cell at a time so partial results are printed / saved
        r = exp.eval_grid('test', [g], stride=a.stride, timing=True)[g]
        res[f'{g[0]}/{g[1]}'] = {k: v for k, v in r.items() if k != 'diag'}
        print(f'[OOD {a.arm} s{a.seed}] L={g[0]:4d} H={g[1]:3d} | windows {r["windows"]} | mse {r["mse"]:.6f} mae {r["mae"]:.6f} | {r["seconds"]:.0f}s | '
              f'per-patch mse ' + ' '.join(f'{v:.4f}' for v in r['patch_mse']), flush=True)
        out = a.out or f'patchmem/logs/weather/ood_{a.arm}_s{a.seed}.json'
        json.dump(dict(arm=a.arm, seed=a.seed, stride=a.stride, train_range=dict(L=[384, 1440], H=[96, 336]), grid=res), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
