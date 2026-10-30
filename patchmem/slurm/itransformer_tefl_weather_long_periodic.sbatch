#!/bin/bash
# More Weather history to supply additional same-clock weekly demonstrations.
#SBATCH --job-name=itr_tefl_w_long_period
#SBATCH --output=patchmem/logs/server/itr_tefl_w_long_period_%j.out
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=04:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

pids=()
for seed in 3047 2024 2025; do
  python -u patchmem/eval_itransformer_tefl_icl_periodic.py \
    --data W --seed "$seed" --histories 4320,5760,10080 --periods 144,1008 \
    --chunk 16 --forecast_bs 16 \
    --out "patchmem/logs/server/itr_tefl_icl/W_long_periodic_s${seed}.json" \
    > "patchmem/logs/server/itr_tefl_icl/W_long_periodic_s${seed}_${SLURM_JOB_ID}.txt" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
