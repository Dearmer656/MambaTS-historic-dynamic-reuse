#!/bin/bash
# Run Weather and full Traffic concurrently in one Slurm GPU allocation.
# Usage: sbatch --dependency=afterok:<ETTh2-job> patchmem/slurm/itransformer_tefl_l96h192_shared.sbatch
#SBATCH --job-name=itr_tefl_shared
#SBATCH --output=patchmem/logs/server/itr_tefl_l96h192_shared_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch

bash patchmem/slurm/itransformer_tefl_l96h192.sbatch W &
weather_pid=$!
bash patchmem/slurm/itransformer_tefl_l96h192.sbatch T &
traffic_pid=$!

weather_status=0
traffic_status=0
wait "$weather_pid" || weather_status=$?
wait "$traffic_pid" || traffic_status=$?
echo "Weather exit=$weather_status Traffic exit=$traffic_status $(date -Is)"
if (( weather_status != 0 || traffic_status != 0 )); then
  exit 1
fi
