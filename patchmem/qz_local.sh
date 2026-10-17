#!/bin/bash
# PAT-304 next-batch experiments 1 and 3 (Linear comment ed09ff1f): where does B2's overfitting come from, the retrieval selection
# (W_Q / W_K) or the stored content (CorrNet C)? Both arms are B2 (Hybrid, lambda_corr 1) with ONE capacity change and NO slot dropout
# (R1_MEM_SLOT_KEEP=1 set explicitly), compared with A1 / B2 / B2-drop50 of the same seed. Two jobs side by side.
#   Q8: R1_MEM_QK_DIM=8   W_Q, W_K: 512 -> 512  =>  512 -> 8 heads x 8 (524,288 -> 65,536 params), same heads / value groups -> DES R1MQ8c1h0s<seed>
#   Z8: R1_CORR_ZDIM=8    CorrNet z_dim 32 -> 8 (Wc (32,32,16) -> (32,8,16); CorrNet 44,112 -> 28,752 params)        -> DES R1MZ8c1h0s<seed>
# usage (repo root): bash patchmem/qz_local.sh [seeds, default 3047]     logs: patchmem/logs/stage3/
# compare: python patchmem/compare_r1.py --a ETTh2_r1_R1h0 --b ETTh2_r1_R1MQ8c1h0 --seeds 3047
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
OUT=patchmem/logs/stage3; mkdir -p $OUT
for s in ${1:-3047}; do
  echo "=== seed $s start $(date '+%F %T')"
  SEED=$s R1_LAMBDA_HIST=0 R1_MEM=1 R1_MEM_SLOT_KEEP=1 R1_MEM_QK_DIM=8 bash patchmem/r1_etth2.sh > $OUT/B2Q8_s$s.txt 2>&1 &
  SEED=$s R1_LAMBDA_HIST=0 R1_MEM=1 R1_MEM_SLOT_KEEP=1 R1_CORR_ZDIM=8 bash patchmem/r1_etth2.sh > $OUT/B2Z8_s$s.txt 2>&1 &
  wait
  echo "=== seed $s done $(date '+%F %T') | Q8: $(grep -c 'R1 test\]' $OUT/B2Q8_s$s.txt) test rows | Z8: $(grep -c 'R1 test\]' $OUT/B2Z8_s$s.txt) test rows"
done
