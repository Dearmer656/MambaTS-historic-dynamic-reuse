#!/bin/bash
# PAT-304 follow-up of the Q8 / Z8 controls (Linear comment ed09ff1f, experiments 1+2 combined with slot dropout). Both arms keep
# slot dropout 0.5 (the only recipe so far that generalises, B2-drop50) and change the retrieval score only; two jobs side by side:
#   Q8+drop50  R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5   low-rank W_Q, W_K (65,536 instead of 524,288)       -> DES R1MK0.5Q8c1h0s<seed>
#   U+drop50   R1_MEM_UNIFORM=1 R1_MEM_SLOT_KEEP=0.5  no W_Q / W_K / tau, a = 1/(kept slots), trained so  -> DES R1MUK0.5c1h0s<seed>
# compare with A1, B2, B2-drop50 (same seed). usage (repo root): bash patchmem/uq_local.sh [seeds, default 3047]   logs: patchmem/logs/stage3/
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
OUT=patchmem/logs/stage3; mkdir -p $OUT
for s in ${1:-3047}; do
  echo "=== seed $s start $(date '+%F %T')"
  SEED=$s R1_LAMBDA_HIST=0 R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 R1_MEM_QK_DIM=8 bash patchmem/r1_etth2.sh > $OUT/B2K50Q8_s$s.txt 2>&1 &
  SEED=$s R1_LAMBDA_HIST=0 R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 R1_MEM_UNIFORM=1 bash patchmem/r1_etth2.sh > $OUT/B2K50U_s$s.txt 2>&1 &
  wait
  echo "=== seed $s done $(date '+%F %T') | Q8+drop50: $(grep -c 'R1 test\]' $OUT/B2K50Q8_s$s.txt) test rows | U+drop50: $(grep -c 'R1 test\]' $OUT/B2K50U_s$s.txt) test rows"
done
