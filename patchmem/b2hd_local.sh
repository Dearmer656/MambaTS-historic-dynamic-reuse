#!/bin/bash
# PAT-304: fixes for the late-rollout harm of B2 (patchmem/diag_rollout.py on the s3047 checkpoints: rollout retrieval helps on train,
# less on val, hurts on test). Two arms side by side (2 jobs), full budget, lambda_hist 0, compared with A1 / B2 of the same seed:
#   B2-H      R1_MEM_ROLL_RETRIEVE=0   retrieval at the history boundaries only            -> DES R1MNc1h0s<seed>
#   B2-drop50 R1_MEM_SLOT_KEEP=0.5     training-time per-sample slot dropout, eval all slots -> DES R1MK0.5c1h0s<seed>
# usage (repo root): bash patchmem/b2hd_local.sh [seeds, default 3047]     logs: patchmem/logs/stage3/
# compare: python patchmem/compare_r1.py --a ETTh2_r1_R1h0 --b ETTh2_r1_R1MNc1h0 --seeds 3047
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
OUT=patchmem/logs/stage3; mkdir -p $OUT
for s in ${1:-3047}; do
  echo "=== seed $s start $(date '+%F %T')"
  SEED=$s R1_LAMBDA_HIST=0 R1_MEM=1 R1_MEM_ROLL_RETRIEVE=0 bash patchmem/r1_etth2.sh > $OUT/B2H_s$s.txt 2>&1 &
  SEED=$s R1_LAMBDA_HIST=0 R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 bash patchmem/r1_etth2.sh > $OUT/B2K50_s$s.txt 2>&1 &
  wait
  echo "=== seed $s done $(date '+%F %T') | B2-H: $(grep -c 'R1 test\]' $OUT/B2H_s$s.txt) test rows | B2-drop50: $(grep -c 'R1 test\]' $OUT/B2K50_s$s.txt) test rows"
done
