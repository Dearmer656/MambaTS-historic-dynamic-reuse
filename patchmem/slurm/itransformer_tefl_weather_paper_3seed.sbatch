#!/bin/bash
# Three independent Weather TEFL paper-protocol runs on one A100.
#SBATCH --job-name=itr_tefl_w_paper
#SBATCH --output=patchmem/logs/server/itr_tefl_w_paper_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch

pids=()
for seed in 3047 2024 2025; do
  bash patchmem/slurm/itransformer_tefl_l96h192.sbatch W "$seed" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
