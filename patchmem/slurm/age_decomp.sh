#!/bin/bash
# PAT-315 shock-age decomposition of the Step 2 carries (eval_sr --age; loaded P / Pfix / Pdiag).  Usage: age_decomp.sh <H>
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ITR_H=$1 ITR_L=720
O=patchmem/logs/server/err_age
mkdir -p $O
for S in 3047 2024 2025; do
  python -u -m patchmem.eval_sr --data T --seed $S --base shrot --freeze_theta --bptt_h 168 --early_stop --val_split \
    --arms s1_P,s1_Pfix,s1_Pdiag --load_carry patchmem/logs/server/err_s2 --age --out_dir $O > $O/T_H$1_L720_s$S.txt 2>&1 || echo "FAIL $1 $S" &
done; wait
