#!/bin/bash
# PAT-317: Timer-XL TimeAttention pos = rope | nope | path | rope_path, official recipe (Table 4 protocol: lookback 672, one model, rolling 96/192/336/720).
# usage: sbatch pat_slurm/p317.sbatch D CFG SEED [EPOCHS]   D = E1 | E2 | W, CFG = a (token 96, N=7) | b (input token 24, N=28; output token 96)
# The 4 arms run in parallel on one GPU. ETTh2 uses the official ETTh1 args (no official ETTh2 script); Weather as PAT-311 repro (Table 11).
#SBATCH --job-name=p317
#SBATCH --output=pat_logs/p317/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm41,elm42,elm74
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/OpenLTM
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=$1; CFG=$2; s=$3; EP=${4:-10}
case $D in
  E1) A="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --e_layers 1 --d_model 1024 --d_ff 2048 --batch_size 32 --learning_rate 0.0001 --lradj type1" ;;
  E2) A="--root_path ./dataset/ETT-small/ --data_path ETTh2.csv --e_layers 1 --d_model 1024 --d_ff 2048 --batch_size 32 --learning_rate 0.0001 --lradj type1" ;;
  W) A="--root_path ./dataset/weather/ --data_path weather.csv --e_layers 4 --d_model 512 --d_ff 2048 --batch_size 32 --learning_rate 0.0005 --lradj type1" ;;
esac
case $CFG in
  a) T="--seq_len 672 --input_token_len 96 --output_token_len 96 --test_seq_len 672" ;;
  b) T="--seq_len 672 --input_token_len 24 --output_token_len 96 --test_seq_len 672" ;;
esac
for P in rope nope path rope_path; do
  (
  ID=p317_${D}_${CFG}_${P}_s${s}_ep${EP}
  C="--task_name forecast --model timer_xl --data MultivariateDatasetBenchmark --model_id $ID --train_epochs $EP --gpu 0 --use_norm --valid_last --num_workers 2 --seed $s --pos $P $T $A"
  LOG=pat_logs/p317/${ID}.txt
  echo "commit $(git rev-parse --short HEAD) $D $CFG $P seed $s ep $EP host $(hostname)" > $LOG
  python -u run.py --is_training 1 --test_pred_len 96 $C >> $LOG 2>&1
  DIR=$(grep -a ">>>>>>>start training" $LOG | tail -1 | sed -E 's/.*training : //; s/>+$//')
  for H in 96 192 336 720; do echo "== H $H" >> $LOG; python -u run.py --is_training 0 --test_pred_len $H --test_dir $DIR $C >> $LOG 2>&1; done
  ) &
done
wait
