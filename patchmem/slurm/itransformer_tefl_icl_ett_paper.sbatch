#!/bin/bash
# Evaluate the corrected TEFL checkpoints on ETTh2, ETTm1, and ETTm2.
#SBATCH --job-name=itr_tefl_icl_ett
#SBATCH --output=patchmem/logs/server/itr_tefl_icl_ett_paper_%j.out
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
for dataset in E M1 M2; do
  python -u patchmem/eval_itransformer_tefl_icl.py \
    --data "$dataset" --seed 3047 --baseline_tag TEFL_paper_L96_E192 \
    --histories 1440,2880 --strides 48,96,144,192,288,384 \
    --out "patchmem/logs/server/itr_tefl_icl/${dataset}_TEFL_paper_L96_E192_s3047.json" \
    > "patchmem/logs/server/itr_tefl_icl/${dataset}_paper_s3047_${SLURM_JOB_ID}.txt" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed != 0 )); then exit "$failed"; fi

env -u PYTHONPATH /home/is/hongyu-s/miniconda3/bin/python patchmem/plot_itransformer_tefl_icl_aggregate.py \
  --datasets E,M1,M2 --seeds 3047 \
  --output_dir patchmem/logs/server/itr_tefl_icl/paper_ett_figures
