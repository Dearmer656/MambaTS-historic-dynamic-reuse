#!/bin/bash
# PAT-304 R1 on ETTh2: one model trained on random lookback 384..1440 / horizon 96..336, tested on the 3x3 grid.
# Same backbone recipe as A0 (e_layers 5, d_model 16, dropout 0.3, lr 1e-3, batch 8, 10 epochs, patience 3, VPT_mode 1 + SA).
# --seq_len/--pred_len only name the setting; the lengths actually used come from R1_* (patchmem/exp_r1.py).
# usage (from the repo root): SEED=3047 R1_LAMBDA_HIST=0.5 bash patchmem/r1_etth2.sh      (works locally or inside sbatch)
SEED=${SEED:-3047}; LAM=${R1_LAMBDA_HIST:-0}
export R1_LAMBDA_HIST=$LAM
PY=${PY:-python}
$PY -u patchmem/run_r1.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv \
  --enc_in 7 --dec_in 7 --c_out 7 --model_id ETTh2_r1 --model MambaTS --data ETTh2 --features M --seed $SEED \
  --seq_len 1440 --label_len 48 --pred_len 336 --e_layers 5 --d_layers 2 --factor 1 --des ${DES:-R1h${LAM}s${SEED}} --itr 1 \
  --n_heads 16 --d_model 16 --dropout 0.3 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA \
  --train_epochs 10 --patience 3 --batch_size 8 --learning_rate 0.001
