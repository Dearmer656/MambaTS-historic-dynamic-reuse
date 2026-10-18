#!/bin/bash
# PAT-304 second dataset: Weather (21 variables), A1 (R1) vs Q8+drop50 (R1Mem, R1_MEM_QK_DIM=8, R1_MEM_SLOT_KEEP=0.5), same recipe for both
# arms (patchmem/r1_weather.sh: e_layers 3, d_model 32, lr 5e-4, batch 8, 10 epochs, patience 3) with a local budget cap of
# R1_MAX_ITERS=1000 iterations per epoch (a full epoch is ~4,390 iterations, ~2.8 h here) -> DES ...I1000.
# Each arm starts when fewer than 2 GPU jobs run; afterwards the PAT-306 training controls (patchmem/sanity_train.sh) are started.
# usage (repo root): bash patchmem/weather_local.sh [seed, default 3047]     logs: patchmem/logs/weather/
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
S=${1:-3047}; OUT=patchmem/logs/weather; mkdir -p $OUT
gpu_jobs() { pgrep -fc "^python -u (patchmem/run_r1|patchmem/run_a0|run|patchmem/sanity_eval)[.]py"; }
for arm in A1 Q8K50; do
  while [ $(gpu_jobs) -ge 2 ]; do sleep 60; done
  echo "=== $arm s$S start $(date '+%F %T')"
  if [ $arm = A1 ]; then
    SEED=$S R1_MAX_ITERS=1000 DES=R1Wh0I1000s$S bash patchmem/r1_weather.sh > $OUT/${arm}_s$S.txt 2>&1 &
  else
    SEED=$S R1_MAX_ITERS=1000 R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 R1_MEM_QK_DIM=8 DES=R1WMK0.5Q8c1h0I1000s$S bash patchmem/r1_weather.sh \
      > $OUT/${arm}_s$S.txt 2>&1 &
  fi
  sleep 90
done
wait
echo "=== done $(date '+%F %T') | A1: $(grep -c 'R1 test\]' $OUT/A1_s$S.txt) test rows | Q8K50: $(grep -c 'R1 test\]' $OUT/Q8K50_s$S.txt) test rows"
nohup bash patchmem/sanity_train.sh > patchmem/logs/sanity/train_queue.txt 2>&1 &
