#!/bin/bash
set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p patchmem/logs/server/itr_icl_supp

run_seed() {
  local seed=$1
  python -u -m patchmem.eval_itransformer_pure_icl \
    --data T --seed "$seed" --cells 576:D1 --clip_w1 0.1 \
    --chunk 4 --forecast_bs 8 \
    --out "patchmem/logs/server/itr_icl_supp/T_pure_full_s${seed}.json" \
    > "patchmem/logs/server/itr_icl_supp/T_pure_full_s${seed}_${SLURM_JOB_ID}.txt" 2>&1
}

run_seed 3047 & first=$!
run_seed 2024 & second=$!
wait "$first"
wait "$second"
run_seed 2025
