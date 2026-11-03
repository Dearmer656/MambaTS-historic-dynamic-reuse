#!/bin/bash
# PAT-315: FreshCast + B test corrections (eval_freshcast --mech_out), Traffic L720 H192, 3 seeds on one GPU.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ITR_H=192 ITR_L=720
O=patchmem/logs/server/err_fc_mech/raw
mkdir -p $O
for S in 3047 2024 2025; do
  python -u -m patchmem.eval_freshcast --data T --seed $S --mech_out $O > $O/T_H192_L720_s$S.txt 2>&1 || echo "FAIL fc $S" &
done; wait
