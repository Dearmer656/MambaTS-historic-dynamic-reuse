#!/bin/bash
# Traffic same-clock-time demonstrations: daily (24) and weekly (168).
#SBATCH --job-name=itr_tefl_t_phase
#SBATCH --output=patchmem/logs/server/itr_tefl_t_phase_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

python -u patchmem/eval_itransformer_tefl_icl_periodic.py \
  --data T --seed 3047 --histories 1440,2880 --periods 24,168 \
  --chunk 2 --forecast_bs 8 \
  --out patchmem/logs/server/itr_tefl_icl/T_periodic_s3047.json \
  > "patchmem/logs/server/itr_tefl_icl/T_periodic_s3047_${SLURM_JOB_ID}.txt" 2>&1
