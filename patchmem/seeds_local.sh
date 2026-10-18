#!/bin/bash
# PAT-304 multi-seed check of the slot-dropout recipe: A1 (R1), B2-drop50 (full W_Q/W_K) and B2-drop50+Q8 on the given seeds, ETTh2,
# lambda_hist 0. Six jobs per two seeds go through a 2-wide queue (max 2 GPU jobs).
# usage (repo root): bash patchmem/seeds_local.sh [seeds, default "3048 3049"]     logs: patchmem/logs/stage3/
# compare: python patchmem/compare_r1.py --a ETTh2_r1_R1h0 --b ETTh2_r1_R1MK0.5Q8c1h0 --seeds 3047,3048,3049 --flag test
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
OUT=patchmem/logs/stage3; mkdir -p $OUT
run() {
  arm=$1; s=$2
  case $arm in
    A1)    env SEED=$s R1_LAMBDA_HIST=0 bash patchmem/r1_etth2.sh ;;
    B2K50) env SEED=$s R1_LAMBDA_HIST=0 R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 bash patchmem/r1_etth2.sh ;;
    Q8K50) env SEED=$s R1_LAMBDA_HIST=0 R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 R1_MEM_QK_DIM=8 bash patchmem/r1_etth2.sh ;;
  esac > $OUT/${arm}_s$s.txt 2>&1
  echo "=== $arm s$s done $(date '+%F %T')"
}
export -f run; export OUT
for s in ${1:-3048 3049}; do for a in A1 B2K50 Q8K50; do echo "$a $s"; done; done | xargs -P2 -L1 bash -c 'run $0 $1'
