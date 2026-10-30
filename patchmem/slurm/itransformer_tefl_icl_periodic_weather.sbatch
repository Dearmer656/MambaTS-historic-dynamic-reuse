#!/bin/bash
# Weather same-clock-time demonstrations: daily (144) and weekly (1008).
#SBATCH --job-name=itr_tefl_w_phase
#SBATCH --output=patchmem/logs/server/itr_tefl_w_phase_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

pids=()
for seed in 3047 2024 2025; do
  python -u patchmem/eval_itransformer_tefl_icl_periodic.py \
    --data W --seed "$seed" --histories 1440,2880 --periods 144,1008 \
    --out "patchmem/logs/server/itr_tefl_icl/W_periodic_s${seed}.json" \
    > "patchmem/logs/server/itr_tefl_icl/W_periodic_s${seed}_${SLURM_JOB_ID}.txt" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
