#!/bin/bash
# Usage: error_path_pack.sh <seed> <demos> <parallel>
# PAT-315 out-of-sample (train on val) A, Bp, Spre, Sperm, Traffic H192 for one seed and demo set, packed onto one GPU.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=${SLURM_JOB_ID:-local}
mkdir -p patchmem/logs/server/error_path
for spec in "A two" "Bp two" "Spre two" "Sperm two"; do echo $spec; done |
  xargs -P "${3:-5}" -L 1 bash -c 'python -u patchmem/eval_error_path.py --seed '"$1"' --demos '"$2"' --variant $0 --beta $1 --train_split val \
    > patchmem/logs/server/error_path/T_H192_'"$2"'_${0}_${1}_trval_s'"$1"'_'"$J"'.txt 2>&1 || echo "FAIL $0 $1 '"$1"'"'
