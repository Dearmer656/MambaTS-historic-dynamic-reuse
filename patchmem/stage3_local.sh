#!/bin/bash
# PAT-304 Stage 3 (comment e18ad84b), local: A1 (R1, memory off) vs B2 (R1Mem Hybrid, lambda_corr 1), ETTh2, paired seeds, full
# budget (10 epochs, patience 3), lambda_hist 0. The two arms of one seed run side by side (2 jobs at a time), seeds one after another.
# Every run ends with the 3x3 val/test grid and, for B2, the same-checkpoint audits (zero / uniform / perm).
# usage (repo root): bash patchmem/stage3_local.sh [seeds, default "3047 3048 3049"]     logs: patchmem/logs/stage3/
# compare: python patchmem/compare_r1.py --a ETTh2_r1_R1h0 --b ETTh2_r1_R1Mc1h0 --seeds 3047,3048,3049
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
OUT=patchmem/logs/stage3; mkdir -p $OUT
for s in ${1:-3047 3048 3049}; do
  echo "=== seed $s start $(date '+%F %T')"
  SEED=$s R1_LAMBDA_HIST=0 bash patchmem/r1_etth2.sh > $OUT/A1_s$s.txt 2>&1 &
  SEED=$s R1_LAMBDA_HIST=0 R1_MEM=1 bash patchmem/r1_etth2.sh > $OUT/B2_s$s.txt 2>&1 &
  wait
  echo "=== seed $s done $(date '+%F %T') | A1: $(grep -c 'R1 test\]' $OUT/A1_s$s.txt) test rows | B2: $(grep -c 'R1 test\]' $OUT/B2_s$s.txt) test rows"
done
