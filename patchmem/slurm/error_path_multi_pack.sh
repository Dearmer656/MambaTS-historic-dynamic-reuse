#!/bin/bash
# Usage: VARS="v1 v2" SEEDS="s1 s2" error_path_multi_pack.sh <demos> <parallel>
# PAT-315, Traffic H192: every (variant, seed) pair packed onto one GPU (beta two).
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=${SLURM_JOB_ID:-local}
mkdir -p patchmem/logs/server/error_path
for s in $SEEDS; do for v in $VARS; do echo "$v $s"; done; done |
  xargs -P "${2:-6}" -L 1 bash -c 'python -u patchmem/eval_error_path.py --seed $1 --demos '"$1"' --variant $0 --beta two \
    > patchmem/logs/server/error_path/T_H192_'"$1"'_${0}_two_s${1}_'"$J"'.txt 2>&1 || echo "FAIL $0 $1"'
