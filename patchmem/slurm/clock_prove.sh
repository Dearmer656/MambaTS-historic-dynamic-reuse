#!/bin/bash
# PAT-315 is the carry's input only a clock?  Arms without input: s1_PFc, s1_PFh, s1_Ph.  Usage: clock_prove.sh <H> [max_c]
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ITR_H=$1 ITR_L=720
V='--data T --base shrot --freeze_theta --bptt_h 168 --early_stop --val_split --dump'
if [ -n "$2" ]; then
  python -u -m patchmem.eval_sr $V --seed 3047 --arms s1_PFc,s1_PFh,s1_Ph --max_c $2 --epochs 2 --out_dir patchmem/logs/server/err_clock_smoke 2>&1 | grep -vE "^\[(carry|stack)" | tail -25
  exit
fi
R=patchmem/logs/server/err_clock
for G in "0 s1_PFc,s1_PFh,s1_Ph" "1000 s1_PFc,s1_PFh,s1_Ph,s1_Pdiag"; do
  set -- $1 $G; CS=$2; ARMS=$3; O=$R/cs$CS; mkdir -p $O
  for S in 3047 2024 2025; do
    python -u -m patchmem.eval_sr $V --seed $S --carry_seed $CS --arms $ARMS --out_dir $O > $O/T_H$1_L720_s$S.txt 2>&1 || echo "FAIL $1 $CS $S" &
  done; wait
done
