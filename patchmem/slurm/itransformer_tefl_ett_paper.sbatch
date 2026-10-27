#!/bin/bash
# ETTh2 for the proposed method; ETTm1/m2 for comparison with TEFL's paper.
# Three datasets share one A100 allocation.
#SBATCH --job-name=itr_tefl_ett_paper
#SBATCH --output=patchmem/logs/server/itr_tefl_ett_paper_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch

pids=()
for dataset in E M1 M2; do
  bash patchmem/slurm/itransformer_tefl_l96h192.sbatch "$dataset" 3047 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
