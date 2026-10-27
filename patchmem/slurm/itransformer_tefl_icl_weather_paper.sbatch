#!/bin/bash
# Run all Weather stride/history cells for three paper-protocol TEFL seeds on one A100.
# Submit only after the baseline reproduction gate has passed.
#SBATCH --job-name=itr_tefl_icl_w
#SBATCH --output=patchmem/logs/server/itr_tefl_icl_w_paper_%j.out
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

mkdir -p patchmem/logs/server/itr_tefl_icl
pids=()
for seed in 3047 2024 2025; do
  python -u patchmem/eval_itransformer_tefl_icl.py \
    --data W --seed "$seed" --baseline_tag TEFL_paper_L96_E192 \
    --histories 1440,2880 --strides 48,96,144,192,288,384 \
    --out "patchmem/logs/server/itr_tefl_icl/W_TEFL_paper_L96_E192_s${seed}.json" \
    > "patchmem/logs/server/itr_tefl_icl/W_paper_s${seed}_${SLURM_JOB_ID}.txt" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed != 0 )); then exit "$failed"; fi

env -u PYTHONPATH /home/is/hongyu-s/miniconda3/bin/python patchmem/plot_itransformer_tefl_icl_aggregate.py \
  --datasets W --seeds 3047,2024,2025
