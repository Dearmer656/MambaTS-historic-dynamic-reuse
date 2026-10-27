#!/bin/bash
# Joint one-step ICL-only (random W1, random W2, g - g0) vs current pure ICL and phase mean, H=192, 5 datasets x 3 seeds.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset ITR_H
J=${SLURM_JOB_ID:-local}
mkdir -p patchmem/logs/server/itr_joint_icl
for D in T W E M1 M2; do for S in 3047 2024 2025; do echo "$D $S"; done; done |
  xargs -P "${1:-8}" -L 1 bash -c 'python -u patchmem/eval_itransformer_joint_icl.py --data $0 --seed $1 > patchmem/logs/server/itr_joint_icl/${0}_s${1}_'"$J"'.txt 2>&1 || echo "FAIL $0 $1"'
