#!/bin/bash
# PAT-307 Timer-XL reproduction (Table 4 / 10 of arXiv 2410.04803: lookback 672, one model, rolling 96/192/336/720).
# ETTh1: official scripts/supervised/rolling_forecast/timer_xl_etth1.sh args. Weather: Table 11 (L4 D512 lr5e-4 bs32 10 ep; lradj type1 as ETTh1,
# the paper does not give the schedule). usage: sbatch pat_slurm/repro.sbatch E1|W|Wc SEED (Wc = Weather with the cosine schedule of the official ECL script)
#SBATCH --job-name=pat307_txl
#SBATCH --output=pat_logs/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/OpenLTM
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=$1; s=$2
case $D in
  E1) A="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1s$s --e_layers 1 --d_model 1024 --d_ff 2048 --batch_size 32 --learning_rate 0.0001 --lradj type1" ;;
  Wc) A="--root_path ./dataset/weather/ --data_path weather.csv --model_id Weathercos$s --e_layers 4 --d_model 512 --d_ff 2048 --batch_size 32 --learning_rate 0.0005 --cosine --tmax 10" ;;
  W) A="--root_path ./dataset/weather/ --data_path weather.csv --model_id Weathers$s --e_layers 4 --d_model 512 --d_ff 2048 --batch_size 32 --learning_rate 0.0005 --lradj type1" ;;
esac
C="--task_name forecast --model timer_xl --data MultivariateDatasetBenchmark --seq_len 672 --input_token_len 96 --output_token_len 96 --test_seq_len 672 --train_epochs 10 --gpu 0 --use_norm --valid_last --seed $s $A"
LOG=pat_logs/repro_${D}_s$s.txt
echo "commit $(git rev-parse --short HEAD) $D seed $s" > $LOG
python -u run.py --is_training 1 --test_pred_len 96 $C >> $LOG 2>&1
DIR=$(grep -a ">>>>>>>start training" $LOG | tail -1 | sed -E 's/.*training : //; s/>+$//')
for H in 96 192 336 720; do echo "== H $H" >> $LOG; python -u run.py --is_training 0 --test_pred_len $H --test_dir $DIR $C >> $LOG 2>&1; done
