#!/bin/bash
# PAT-306 baseline sanity audit, training controls (seed 3047 first). Each job starts when fewer than 2 GPU jobs (training or sanity
# eval) are running; jobs in priority order:
#   T1   A0 (original MambaTS, direct head) L=1440 H=96, paper H=96 config            -> is A1's degradation at L=1440 an R1 bug?
#   T3   Q8+drop50 with CorrNet residuals rolled across the batch (R1_CORR_SHUFFLE=1)  -> gain from information or from the extra path?
#   T2   A1 with d_model 26 (315,873 params ~ Q8 model 315,400; A1 189,433)          -> gain from capacity?
#   T4a  A1 trained on fixed L=720 / H=96 (R1_LMIN=LMAX=720, R1_HMIN=HMAX=2), val 720:96 -> gain only in the one-model-for-all setting?
#   T4b  Q8+drop50, same fixed L=720 / H=96
#   T5a  A1 lr 5e-4                                                                   -> baseline left untuned?
#   T5b  A1 15 epochs, patience 5
# usage (repo root): bash patchmem/sanity_train.sh [seed, default 3047] [jobs, default "T1 T3 T2 T4a T4b T5a T5b"]   logs: patchmem/logs/sanity/
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
S=${1:-3047}; OUT=patchmem/logs/sanity; mkdir -p $OUT
R1="--task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 --dec_in 7 --c_out 7
 --model_id ETTh2_r1 --model MambaTS --data ETTh2 --features M --seed $S --seq_len 1440 --label_len 48 --pred_len 336 --e_layers 5
 --d_layers 2 --factor 1 --itr 1 --n_heads 16 --dropout 0.3 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --batch_size 8"
Q8="R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 R1_MEM_QK_DIM=8"
FIX="R1_LMIN=720 R1_LMAX=720 R1_HMIN=2 R1_HMAX=2 R1_VAL_GRID=720:96"
job() {
  case $1 in
    T1)  python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 \
           --dec_in 7 --c_out 7 --model_id ETTh2_1440_96 --model MambaTS --data ETTh2 --features M --seq_len 1440 --label_len 48 \
           --pred_len 96 --e_layers 5 --d_layers 2 --factor 1 --des A0L1440s$S --itr 1 --n_heads 16 --d_model 16 --dropout 0.3 \
           --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --train_epochs 10 --patience 3 --batch_size 8 --learning_rate 0.001 --seed $S ;;
    T3)  env $Q8 R1_CORR_SHUFFLE=1 python -u patchmem/run_r1.py $R1 --d_model 16 --des R1MSK0.5Q8c1h0s$S --train_epochs 10 --patience 3 --learning_rate 0.001 ;;
    T2)  python -u patchmem/run_r1.py $R1 --d_model 26 --des R1h0dm26s$S --train_epochs 10 --patience 3 --learning_rate 0.001 ;;
    T4a) env $FIX python -u patchmem/run_r1.py $R1 --d_model 16 --des R1h0F720x96s$S --train_epochs 10 --patience 3 --learning_rate 0.001 ;;
    T4b) env $FIX $Q8 python -u patchmem/run_r1.py $R1 --d_model 16 --des R1MK0.5Q8c1h0F720x96s$S --train_epochs 10 --patience 3 --learning_rate 0.001 ;;
    T5a) python -u patchmem/run_r1.py $R1 --d_model 16 --des R1h0lr5e-4s$S --train_epochs 10 --patience 3 --learning_rate 0.0005 ;;
    T5b) python -u patchmem/run_r1.py $R1 --d_model 16 --des R1h0e15p5s$S --train_epochs 15 --patience 5 --learning_rate 0.001 ;;
  esac > $OUT/${1}_s$S.txt 2>&1
  echo "=== $1 s$S done $(date '+%F %T') | test rows $(grep -c 'R1 test\]\|^mse:' $OUT/${1}_s$S.txt)"
}
for j in ${2:-T1 T3 T2 T4a T4b T5a T5b}; do
  while [ $(pgrep -fc "^python -u (patchmem/run_r1|run|patchmem/sanity_eval)[.]py") -ge 2 ]; do sleep 60; done
  echo "=== $j s$S start $(date '+%F %T')"
  job $j &
  sleep 90
done
wait
echo "=== all done $(date '+%F %T')"
