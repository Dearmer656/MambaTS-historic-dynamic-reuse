#!/bin/bash
# PAT-304 third dataset: Traffic-32 (32 random sensors, patchmem/make_traffic32.py), A1 (R1) vs Q8+drop50 (R1Mem, R1_MEM_QK_DIM=8,
# R1_MEM_SLOT_KEEP=0.5), same recipe for both arms (patchmem/r1_traffic32.sh = the Weather recipe), full budget (no iteration cap:
# 10,505 training windows, ~1,313 iterations per epoch). Each arm starts when fewer than 2 GPU jobs run; afterwards the PAT-306 training
# controls (patchmem/sanity_train.sh) are started.
# usage (repo root): bash patchmem/traffic32_local.sh [seed, default 3047]     logs: patchmem/logs/traffic32/
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
S=${1:-3047}; OUT=patchmem/logs/traffic32; mkdir -p $OUT
gpu_jobs() { pgrep -fc "^python -u (patchmem/run_r1|patchmem/run_a0|run|patchmem/sanity_eval|patchmem/ood_eval)[.]py"; }
for arm in A1 Q8K50; do
  while [ $(gpu_jobs) -ge 2 ]; do sleep 60; done
  echo "=== $arm s$S start $(date '+%F %T')"
  if [ $arm = A1 ]; then
    SEED=$S DES=R1Th0s$S bash patchmem/r1_traffic32.sh > $OUT/${arm}_s$S.txt 2>&1 &
  else
    SEED=$S R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 R1_MEM_QK_DIM=8 DES=R1TMK0.5Q8c1h0s$S bash patchmem/r1_traffic32.sh > $OUT/${arm}_s$S.txt 2>&1 &
  fi
  sleep 90
done
wait
echo "=== traffic32 done $(date '+%F %T')"
nohup bash patchmem/sanity_train.sh > patchmem/logs/sanity/train_queue.txt 2>&1 &
