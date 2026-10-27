#!/bin/bash
# Pack all phase-control runs (datasets x seeds) onto one GPU in parallel.  Args: backbone (tefl|base)
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
B=${1:-base}; J=${SLURM_JOB_ID:-local}
mkdir -p patchmem/logs/server/itr_phase_controls
for c in "T 2880 168" "E 1440 24" "M1 2880 672" "M2 2880 96" "W 2880 144"; do
  set -- $c
  for S in 3047 2024 2025; do
    python -u patchmem/eval_itransformer_tefl_phase_controls.py --data $1 --seed $S --history $2 --period $3 --backbone $B \
      > "patchmem/logs/server/itr_phase_controls/${1}_N${2}_P${3}_s${S}_${B}_${J}.txt" 2>&1 &
  done
done
wait
