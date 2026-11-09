#!/bin/bash
# Usage: itransformer_base_longh_pack.sh "<datasets>" <parallel> ["<horizons>", default "336 720"]
# Standalone iTransformer L96, H in {336, 720}, seeds 3047/2024/2025, packed onto one GPU (xargs -P).
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=${SLURM_JOB_ID:-local}
for H in ${3:-336 720}; do for D in $1; do for S in 3047 2024 2025; do echo "$D $S $H"; done; done; done |
  xargs -P "${2:-4}" -L 1 bash -c 'python -u -m patchmem.run_itransformer_base --data $0 --seed $1 --pred_len $2 > patchmem/logs/server/itr_base_${0}_H${2}_s${1}_'"$J"'.txt 2>&1 || echo "FAIL $0 $1 $2"'
