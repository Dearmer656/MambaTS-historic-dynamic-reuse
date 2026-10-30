#!/bin/bash
# Pack all phase-control runs (datasets x seeds) onto one GPU in parallel.  Sequential bias -> W2 -> W1 GD on the TEFL head (day vs week, n=4,8,16).
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=${SLURM_JOB_ID:-local}
mkdir -p patchmem/logs/server/itr_twolayer
for c in "T 2880 168" "E 1440 24" "M1 2880 672" "M2 2880 96" "W 2880 144"; do
  set -- $c
  for S in 3047 2024 2025; do
    python -u patchmem/eval_itransformer_tefl_twolayer.py --data $1 --seed $S \
      > "patchmem/logs/server/itr_twolayer/${1}_s${S}_${J}.txt" 2>&1 &
  done
done
wait
