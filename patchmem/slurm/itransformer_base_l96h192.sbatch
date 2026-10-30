#!/bin/bash
# Usage: sbatch patchmem/slurm/itransformer_base_l96h192.sbatch E|M1|M2|W|T
# Three standalone iTransformer seeds, sharing one A100 where memory permits.
#SBATCH --job-name=itr_base_96_192
#SBATCH --output=patchmem/logs/server/itr_base_96_192_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

dataset=${1:?Specify E, M1, M2, W, or T}
case "$dataset" in E|M1|M2|W|T) ;; *) exit 2 ;; esac

run_seed() {
  local seed=$1
  python -u -m patchmem.run_itransformer_base --data "$dataset" --seed "$seed" \
    > "patchmem/logs/server/itr_base_${dataset}_s${seed}_${SLURM_JOB_ID}.txt" 2>&1
}

pids=()
for seed in 3047 2024; do run_seed "$seed" & pids+=("$!"); done
if [[ "$dataset" != T ]]; then run_seed 2025 & pids+=("$!"); fi
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed != 0 )); then exit "$failed"; fi
if [[ "$dataset" == T ]]; then run_seed 2025; fi
