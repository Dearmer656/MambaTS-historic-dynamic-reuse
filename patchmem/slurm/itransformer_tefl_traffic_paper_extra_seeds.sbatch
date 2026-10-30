#!/bin/bash
# Two additional Traffic seeds; training and evaluation share one A100.
#SBATCH --job-name=itr_tefl_t_3seed
#SBATCH --output=patchmem/logs/server/itr_tefl_t_3seed_%j.out
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

pids=()
for seed in 2024 2025; do
  bash patchmem/slurm/itransformer_tefl_l96h192.sbatch T "$seed" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed != 0 )); then exit "$failed"; fi

run_seed() {
  local seed=$1
  python -u patchmem/eval_itransformer_tefl_icl.py \
    --data T --seed "$seed" --baseline_tag TEFL_paper_L96_E192 \
    --histories 1440,2880 --strides 48,96,144,192,288,384 \
    --chunk 2 --forecast_bs 8 \
    --out "patchmem/logs/server/itr_tefl_icl/T_TEFL_paper_L96_E192_s${seed}.json" \
    > "patchmem/logs/server/itr_tefl_icl/T_paper_s${seed}_${SLURM_JOB_ID}.txt" 2>&1
  python -u patchmem/eval_itransformer_tefl_icl_periodic.py \
    --data T --seed "$seed" --histories 1440,2880 --periods 24,168 \
    --chunk 2 --forecast_bs 8 \
    --out "patchmem/logs/server/itr_tefl_icl/T_periodic_s${seed}.json" \
    > "patchmem/logs/server/itr_tefl_icl/T_periodic_s${seed}_${SLURM_JOB_ID}.txt" 2>&1
}

pids=()
for seed in 2024 2025; do run_seed "$seed" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed != 0 )); then exit "$failed"; fi

env -u PYTHONPATH /home/is/hongyu-s/miniconda3/bin/python \
  patchmem/plot_itransformer_tefl_icl_periodic.py --datasets W,T
