#!/bin/bash
# PAT-315 lead-wise stacking: (1) raw FC / B corrections on every val+test origin, (2) eval_sr stacking of
# [S_e, P (Step 2 carry, loaded), FC, B] per lead bucket on val_fit; Traffic L720 H192, 3 seeds on one GPU.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ITR_H=192 ITR_L=720
O=patchmem/logs/server/err_stack
mkdir -p $O/fc
for S in 3047 2024 2025; do
  { [ -f $O/fc/T_H192_L720_s${S}_ext.npz ] || python -u -m patchmem.eval_freshcast --data T --seed $S --ext_out $O/fc > $O/fc/T_H192_L720_s$S.txt 2>&1 || echo "FAIL fc $S"
    python -u -m patchmem.eval_sr --data T --seed $S --base shrot --freeze_theta --bptt_h 168 --early_stop --val_split --arms s1_P \
      --load_carry patchmem/logs/server/err_s2 --ext FC,B --ext_dir $O/fc --out_dir $O > $O/T_H192_L720_s$S.txt 2>&1 || echo "FAIL sr $S"; } &
done; wait
