#!/bin/bash
# PAT-304 null slot (R1_MEM_NULL=1, a value-0 slot in every head's softmax: retrieval can abstain when no past slot matches) on top of the
# current best recipe Q8 + slot dropout 0.5 -> DES R1MK0.5Q8Oc1h0s<seed>; compare with Q8+drop50 (R1MK0.5Q8c1h0) and A1 on the same seeds.
# Waits for any running seeds_local.sh queue first (max 2 GPU jobs); the given seeds run side by side.
# usage (repo root): bash patchmem/null_local.sh [seeds, default "3047 3048"]     logs: patchmem/logs/stage3/
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
OUT=patchmem/logs/stage3; mkdir -p $OUT
while pgrep -f "bash patchmem/seeds_local.sh" > /dev/null; do sleep 120; done
echo "=== start $(date '+%F %T')"
for s in ${1:-3047 3048}; do
  SEED=$s R1_LAMBDA_HIST=0 R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 R1_MEM_QK_DIM=8 R1_MEM_NULL=1 bash patchmem/r1_etth2.sh > $OUT/Q8K50O_s$s.txt 2>&1 &
done
wait
echo "=== done $(date '+%F %T') | $(for s in ${1:-3047 3048}; do echo -n "s$s: $(grep -c 'R1 test\]' $OUT/Q8K50O_s$s.txt) test rows  "; done)"
