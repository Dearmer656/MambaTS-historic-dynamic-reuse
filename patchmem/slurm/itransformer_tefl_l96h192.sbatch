#!/bin/bash
# Usage: sbatch patchmem/slurm/itransformer_tefl_l96h192.sbatch E|M1|M2|W|T [seed]
# TEFL paper protocol: 3 ordered warm-up epochs (MAE + batch-axis spectral flatness),
# 12 joint epochs (AdamW), 192-step error window, rank 64.
# E=ETTh2, M1=ETTm1, M2=ETTm2, W=Weather, T=full Traffic (862 sensors).
#SBATCH --job-name=itr_tefl_96_192
#SBATCH --output=patchmem/logs/server/itr_tefl_l96h192_%x_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TEFL_PAPER_TRAIN=1

dataset=${1:?Specify E, W, or T}
seed=${2:-3047}
case "$dataset" in
  E) name=ETTh2; data=ETTh2; root=./dataset/ETT-small; file=ETTh2.csv; channels=7
     layers=2; dim=128; ff=128; batch=32; lr=0.0001 ;;
  M1) name=ETTm1; data=ETTm1; root=./dataset/ETT-small; file=ETTm1.csv; channels=7
     layers=2; dim=128; ff=128; batch=32; lr=0.0001 ;;
  M2) name=ETTm2; data=ETTm2; root=./dataset/ETT-small; file=ETTm2.csv; channels=7
     layers=2; dim=128; ff=128; batch=32; lr=0.0001 ;;
  W) name=weather; data=custom; root=./dataset/weather; file=weather.csv; channels=21
     layers=3; dim=512; ff=512; batch=32; lr=0.0001 ;;
  T) name=traffic; data=custom; root=./dataset/traffic; file=traffic.csv; channels=862
     layers=4; dim=512; ff=512; batch=16; lr=0.001 ;;
  *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac

echo "start $(date -Is) job=$SLURM_JOB_ID host=$(hostname) dataset=$name git=$(git rev-parse --short HEAD)"
run_log="patchmem/logs/server/itr_tefl_l96h192_${dataset}_paper_s${seed}_${SLURM_JOB_ID}.txt"
python -u patchmem/run_itransformer_tefl.py \
  --task_name long_term_forecast --is_training 1 \
  --model iTransformer --model_id "${name}_96_192_TEFL" \
  --data "$data" --root_path "$root" --data_path "$file" --features M \
  --seq_len 96 --label_len 48 --pred_len 192 \
  --enc_in "$channels" --dec_in "$channels" --c_out "$channels" \
  --e_layers "$layers" --d_model "$dim" --d_ff "$ff" \
  --batch_size "$batch" --learning_rate "$lr" \
  --VPT_mode 0 --patience 3 --loss L1 --seed "$seed" --itr 1 --des "TEFL_paper_L96_E192_s${seed}" \
  --checkpoints ./checkpoints --results ./results \
  > "$run_log" 2>&1
echo "end $(date -Is) dataset=$name"
