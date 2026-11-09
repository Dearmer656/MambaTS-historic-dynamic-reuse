#!/bin/bash
# PAT-315 causal rolling stack (eval_sr --roll): coefficients re-fit at every origin on the last W matured val/test
# origins; static (val_fit) stack printed alongside.  Ext: FC, B (eval_freshcast) + pathret v1 arms.  Usage: stack_roll.sh [max_c]
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ITR_H=192 ITR_L=720
O=patchmem/logs/server/err_stack_roll; PR=patchmem/logs/server/err_pathret
MC=""; SEEDS="3047 2024 2025"; [ -n "$1" ] && { MC="--max_c $1"; O=${O}_smoke; SEEDS=3047; }
mkdir -p $O
for S in $SEEDS; do
  python -u -m patchmem.eval_sr --data T --seed $S --base shrot --freeze_theta --bptt_h 168 --early_stop --val_split --arms s1_P $MC \
    --load_carry patchmem/logs/server/err_s2 --ext FC,B,B0@$PR/B0,A1@$PR/A1,A2@$PR/A2,A2s@$PR/A2s --ext_dir patchmem/logs/server/err_stack/fc \
    --roll 168,336,672 --out_dir $O > $O/T_H192_L720_s$S.txt 2>&1 || echo "FAIL sr $S" &
done; wait
