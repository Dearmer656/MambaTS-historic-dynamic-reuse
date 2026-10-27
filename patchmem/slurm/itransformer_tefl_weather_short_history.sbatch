#!/bin/bash
# Test whether fewer same-clock-time Weather demonstrations improve adaptation.
#SBATCH --job-name=itr_tefl_w_short
#SBATCH --output=patchmem/logs/server/itr_tefl_w_short_%j.out
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=04:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

pids=()
for seed in 3047 2024 2025; do
  python -u patchmem/eval_itransformer_tefl_icl_periodic.py \
    --data W --seed "$seed" --histories 576,720,1008 --periods 144 \
    --chunk 16 --forecast_bs 16 \
    --out "patchmem/logs/server/itr_tefl_icl/W_short_history_s${seed}.json" \
    > "patchmem/logs/server/itr_tefl_icl/W_short_history_s${seed}_${SLURM_JOB_ID}.txt" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
