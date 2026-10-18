#!/bin/bash
# PAT-304 Weather: lookback / horizon outside the training range (patchmem/ood_eval.py), A1 then Q8+drop50, seed 3047. Each arm starts when fewer
# than 2 GPU jobs run. usage (repo root): bash patchmem/ood_local.sh [seed]     logs: patchmem/logs/weather/ood_<arm>_s<seed>.txt
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
S=${1:-3047}; OUT=patchmem/logs/weather
gpu_jobs() { pgrep -fc "^python -u (patchmem/run_r1|patchmem/run_a0|run|patchmem/sanity_eval|patchmem/ood_eval)[.]py"; }
for arm in A1 Q8; do
  while [ $(gpu_jobs) -ge 2 ]; do sleep 60; done
  echo "=== $arm s$S start $(date '+%F %T')"
  R1_AUDIT=0 python -u patchmem/ood_eval.py --arm $arm --seed $S > $OUT/ood_${arm}_s$S.txt 2>&1
done
