#!/bin/bash
# PAT-304 R1 on ETTh2: one model trained on random lookback 384..1440 / horizon 96..336, tested on the 3x3 grid.
# Same backbone recipe as A0 (e_layers 5, d_model 16, dropout 0.3, lr 1e-3, batch 8, 10 epochs, patience 3, VPT_mode 1 + SA).
# --seq_len/--pred_len only name the setting; the lengths actually used come from R1_* (patchmem/exp_r1.py).
# usage (from the repo root): SEED=3047 R1_LAMBDA_HIST=0.5 bash patchmem/r1_etth2.sh      (works locally or inside sbatch)
# EXTRA: run.py arguments appended last (they override the ones above), e.g. EXTRA="--d_model 26"
# smoke: EPOCHS=1 R1_MAX_ITERS=200 R1_VAL_STRIDE=16 DES=R1smoke bash patchmem/r1_etth2.sh
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
# R1_OBS=1 (observation-space error memory, patchmem/r1_obsmem.py): Ob + U (uniform) + K<p> + Q<r> + F0 (backbone patch fed back), e.g. ObK0.5Q8
[ "${R1_OBS:-0}" = 1 ] && TAG=Ob$([ "${R1_MEM_UNIFORM:-0}" = 1 ] && echo U)$([ "${R1_MEM_SLOT_KEEP:-1}" = 1 ] || echo K${R1_MEM_SLOT_KEEP})Q${R1_MEM_QK_DIM:-8}$([ "${R1_OBS_FEED:-1}" = 1 ] || echo F0)$([ "${R1_OBS_PLAIN:-0}" = 1 ] && echo Pl)
# R1_TEFL=native|faithful (TEFL-style error feedback, patchmem/r1_tefl.py, PAT-307): Tf + n|f + H<Hc> + R<rank> + F0, e.g. TfnH2R64
# R1_PATHRES=1 (path-transported residuals, patchmem/r1_pathres.py, PAT-307): Pr + I (H = I) + R<rank> + K<p>, e.g. PrR32
[ "${R1_PATHRES:-0}" = 1 ] && TAG=Pr$([ "${R1_PATHRES_NOH:-0}" = 1 ] && echo I)R${R1_PATHRES_RANK:-32}$([ "${R1_MEM_SLOT_KEEP:-1}" = 1 ] || echo K${R1_MEM_SLOT_KEEP})
[ -n "${R1_TEFL}" ] && TAG=Tf${R1_TEFL:0:1}H${R1_TEFL_HC:-2}R${R1_TEFL_RANK:-64}$([ "${R1_TEFL_FEED:-1}" = 1 ] || echo F0)
# R1_MIX=1 (per-sample random lookback / horizon in every batch): suffix mx, e.g. R1WMK0.5Q8c1mxh0s3047
[ "${R1_MIX:-0}" = 1 ] && TAG=${TAG}mx
PY=${PY:-python}
$PY -u patchmem/run_r1.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv \
  --enc_in 7 --dec_in 7 --c_out 7 --model_id ETTh2_r1 --model MambaTS --data ETTh2 --features M --seed $SEED \
  --seq_len 1440 --label_len 48 --pred_len 336 --e_layers 5 --d_layers 2 --factor 1 --des ${DES:-R1${TAG}h${LAM}s${SEED}} --itr 1 \
  --n_heads 16 --d_model 16 --dropout 0.3 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA \
  --train_epochs ${EPOCHS:-10} --patience 3 --batch_size 8 --learning_rate 0.001 ${EXTRA}
