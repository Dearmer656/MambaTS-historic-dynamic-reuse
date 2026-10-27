#!/bin/bash
# Usage: itransformer_stepsize_pack.sh <parallel>
# Step-size rules (cos / nlms / const) for the W2 -> W1 case-wise step, TEFL and standalone backbones, H=192.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=${SLURM_JOB_ID:-local}
mkdir -p patchmem/logs/server/itr_stepsize
for B in tefl base; do for D in T W E M1 M2; do for S in 3047 2024 2025; do echo "$D $S $B"; done; done; done |
  xargs -P "${1:-10}" -L 1 bash -c 'python -u patchmem/eval_itransformer_stepsize.py --data $0 --seed $1 --backbone $2 > patchmem/logs/server/itr_stepsize/${0}_s${1}_${2}_'"$J"'.txt 2>&1 || echo "FAIL $0 $1 $2"'
