#!/bin/bash
#SBATCH --job-name=itr_T_pure_day
#SBATCH --output=patchmem/logs/server/itr_icl_supp_%j.out
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p patchmem/logs/server/itr_icl_supp

run_seed() {
  local seed=$1
  python -u -m patchmem.eval_itransformer_pure_icl \
    --data T --seed "$seed" --cells 1440:P24,2880:P24 --clip_w1 0.1 \
    --chunk 4 --forecast_bs 8 \
    --out "patchmem/logs/server/itr_icl_supp/T_pure_day_s${seed}.json" \
    > "patchmem/logs/server/itr_icl_supp/T_pure_day_s${seed}_${SLURM_JOB_ID}.txt" 2>&1
}

run_seed 3047 & first=$!
run_seed 2024 & second=$!
wait "$first"
wait "$second"
run_seed 2025
