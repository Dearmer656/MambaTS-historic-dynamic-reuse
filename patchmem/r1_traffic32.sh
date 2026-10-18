#!/bin/bash
# PAT-304 R1 on Traffic-32 (32 random sensors of Traffic, patchmem/make_traffic32.py): one model trained on random lookback 384..1440 / horizon 96..336, tested on the 3x3 grid.
# Same backbone recipe as patchmem/r1_weather.sh (Weather, 21 variables, the closest channel count; the official Traffic script uses patch_len = seq_len, not usable here) (e_layers 3, d_model 32, d_ff 512,
# dropout 0.2, lr 5e-4, batch 8, 10 epochs, patience 3, VPT_mode 1 + SA). Mirrors patchmem/r1_etth2.sh.
# --seq_len/--pred_len only name the setting; the lengths actually used come from R1_* (patchmem/exp_r1.py).
# usage (from the repo root): SEED=3047 R1_LAMBDA_HIST=0.5 bash patchmem/r1_traffic32.sh
# smoke: EPOCHS=1 R1_MAX_ITERS=100 R1_VAL_STRIDE=64 DES=R1Tsmoke bash patchmem/r1_traffic32.sh
# memory: R1_MEM=1 (R1Mem, patchmem/r1_mem.py; see exp_r1.py for R1_CORR / R1_MEM_GROUP / R1_CORR_SHUFFLE / R1_LAMBDA_CORR /
# R1_MEM_VALUE_DETACH / R1_MEM_RETRIEVE). The default DES gets a tag per arm so no two arms share a checkpoint dir:
# M (memory) + S (corr shuffle) + D (values detached, Pure-B) + X (retrieval off, B3) + N (no rollout retrieval, B2-H)
# + U (uniform attention trained from scratch, no W_Q / W_K) + K<p> (training slot keep probability, B2-drop) + Q<r> (per-head q/k dim) + Z<z> (CorrNet z_dim) + O (null slot) + c<lambda_corr>.
#   A1 baseline: (none) | B0: R1_MEM=1 R1_LAMBDA_CORR=0 -> Mc0 | B1: R1_MEM=1 R1_MEM_VALUE_DETACH=1 -> MDc1
#   B2 Hybrid: R1_MEM=1 -> Mc1 | B3: R1_MEM=1 R1_MEM_RETRIEVE=0 -> MXc1
#   B2-H: R1_MEM=1 R1_MEM_ROLL_RETRIEVE=0 -> MNc1 | B2-drop50: R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 -> MK0.5c1
#   low-rank Q/K: R1_MEM=1 R1_MEM_QK_DIM=8 -> MQ8c1 | small C code: R1_MEM=1 R1_CORR_ZDIM=8 -> MZ8c1
#   selection-free: R1_MEM=1 R1_MEM_UNIFORM=1 -> MUc1 | combined with slot dropout: ... R1_MEM_SLOT_KEEP=0.5 -> MUK0.5c1, with low-rank: MK0.5Q8c1
SEED=${SEED:-3047}; LAM=${R1_LAMBDA_HIST:-0}
export R1_LAMBDA_HIST=$LAM
TAG=$([ "${R1_MEM:-0}" = 1 ] && echo M$([ "${R1_CORR_SHUFFLE:-0}" = 1 ] && echo S)$([ "${R1_MEM_VALUE_DETACH:-0}" = 1 ] && echo D)$([ "${R1_MEM_RETRIEVE:-1}" = 1 ] || echo X)$([ "${R1_MEM_ROLL_RETRIEVE:-1}" = 1 ] || echo N)$([ "${R1_MEM_UNIFORM:-0}" = 1 ] && echo U)$([ "${R1_MEM_SLOT_KEEP:-1}" = 1 ] || echo K${R1_MEM_SLOT_KEEP})$([ -z "${R1_MEM_QK_DIM}" ] || echo Q${R1_MEM_QK_DIM})$([ "${R1_CORR_ZDIM:-32}" = 32 ] || echo Z${R1_CORR_ZDIM})$([ "${R1_MEM_NULL:-0}" = 1 ] && echo O)c${R1_LAMBDA_CORR:-1})
PY=${PY:-python}
$PY -u patchmem/run_r1.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/traffic32 --data_path traffic32.csv --target 856 \
  --enc_in 32 --dec_in 32 --c_out 32 --model_id traffic32_r1 --model MambaTS --data custom --features M --seed $SEED \
  --seq_len 1440 --label_len 48 --pred_len 336 --e_layers 3 --d_layers 2 --factor 1 --des ${DES:-R1T${TAG}h${LAM}s${SEED}} --itr 1 \
  --n_heads 16 --d_model 32 --d_ff 512 --dropout 0.2 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA \
  --train_epochs ${EPOCHS:-10} --patience 3 --batch_size 8 --learning_rate 0.0005
