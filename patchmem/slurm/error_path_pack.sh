#!/bin/bash
# Usage: error_path_pack.sh <seed> <parallel>
# PAT-315 stage 1, Traffic H192: 7 variants x {dense, phase} demos for one seed, packed onto one GPU.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=${SLURM_JOB_ID:-local}
mkdir -p patchmem/logs/server/error_path
for K in dense phase; do for V in C Crev Cerr End Bp B A; do echo "$K $V"; done; done |
  xargs -P "${2:-5}" -L 1 bash -c 'python -u patchmem/eval_error_path.py --seed '"$1"' --demos $0 --variant $1 > patchmem/logs/server/error_path/T_H192_${0}_${1}_s'"$1"'_'"$J"'.txt 2>&1 || echo "FAIL $0 $1 '"$1"'"'
