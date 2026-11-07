#!/bin/bash
# PAT-315 per-cell gain maps (eval_sr --src) of Pfix cs1000 and PFw cs0 / cs1000, to compare across carry seeds.  Usage: src_seed.sh <H>
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ITR_H=$1 ITR_L=720
V='--data T --base shrot --freeze_theta --bptt_h 168 --early_stop --val_split --src'
L=patchmem/logs/server
for G in "s1_Pfix $L/err_gate/cs1000 $L/err_src_seed/Pfix_cs1000" "s1_PFw $L/err_srcx/cs0 $L/err_src_seed/PFw_cs0" "s1_PFw $L/err_srcx/cs1000 $L/err_src_seed/PFw_cs1000"; do
  set -- $1 $G; A=$2; LC=$3; O=$4; mkdir -p $O
  for S in 3047 2024 2025; do
    python -u -m patchmem.eval_sr $V --seed $S --arms $A --load_carry $LC --out_dir $O > $O/T_H$1_L720_s$S.txt 2>&1 || echo "FAIL $1 $A $S" &
  done; wait
done
