#!/bin/bash
# PAT-304 one-model-for-all-horizons control for the original MambaTS (A0): train ONCE at L=720, pred_len=336 (run.py, direct head), then
# evaluate by truncation on the H=96/192/336 test sets (patchmem/a0_trunc_eval.py). Two configs side by side:
#   R1cfg    the R1 / R1Mem hyper-parameters (= paper's H=96 config: el 5, dm 16, bs 8, lr 1e-3)      des A0T336R1cfg
#   paper336 the paper's own H=336 config (el 2, dm 32, bs 32, lr 5e-4)                               des A0T336paper
# Waits for a running seeds_local.sh queue (max 2 GPU jobs); afterwards starts null_local.sh.
# usage (repo root): bash patchmem/a0trunc_local.sh [seed, default 3047]     logs: patchmem/logs/a0trunc/
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
OUT=patchmem/logs/a0trunc; mkdir -p $OUT
S=${1:-3047}
while pgrep -f "bash patchmem/seeds_local.sh" > /dev/null; do sleep 120; done
echo "=== start $(date '+%F %T')"
BASE="--task_name long_term_forecast --root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 --dec_in 7 --c_out 7
 --model_id ETTh2_720_336 --model MambaTS --data ETTh2 --features M --seq_len 720 --label_len 48 --pred_len 336 --d_layers 2 --factor 1
 --itr 1 --n_heads 16 --dropout 0.3 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --train_epochs 10 --patience 3 --seed $S"
declare -A CFG=([R1cfg]="--e_layers 5 --d_model 16 --batch_size 8 --learning_rate 0.001" [paper336]="--e_layers 2 --d_model 32 --batch_size 32 --learning_rate 0.0005")
for k in R1cfg paper336; do
  (python -u run.py --is_training 1 $BASE ${CFG[$k]} --des A0T336${k}s$S > $OUT/train_$k.txt 2>&1
   setting=$(grep -o 'start training : [^>]*' $OUT/train_$k.txt | sed 's/start training : //')
   python -u patchmem/a0_trunc_eval.py --setting $setting --H 96,192,336 --out $OUT/trunc_${k}_s$S.json -- --is_training 0 $BASE ${CFG[$k]} \
     --des A0T336${k}s$S > $OUT/eval_$k.txt 2>&1) &
done
wait
echo "=== done $(date '+%F %T')"; grep -h "A0 trunc\]" $OUT/eval_*.txt
nohup bash patchmem/null_local.sh > patchmem/logs/stage3/null_queue.txt 2>&1 &
