#!/bin/bash
# Every causally eligible history origin is a TEFL-error demonstration.
#SBATCH --job-name=itr_tefl_w_full
#SBATCH --output=patchmem/logs/server/itr_tefl_w_full_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

pids=()
for seed in 3047 2024 2025; do
  python -u patchmem/eval_itransformer_tefl_icl.py \
    --data W --seed "$seed" --histories 576,720,1008,1440,2880 \
    --strides 1 --chunk 4 --forecast_bs 16 \
    --baseline_tag TEFL_paper_L96_E192 \
    --out "patchmem/logs/server/itr_tefl_icl/W_full_lookback_s${seed}.json" \
    > "patchmem/logs/server/itr_tefl_icl/W_full_lookback_s${seed}_${SLURM_JOB_ID}.txt" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
