#!/bin/bash
# PAT-304 server queue: the runs the laptop cannot afford. Same recipe for both arms everywhere (no tuning of either arm).
#   W  Weather FULL budget (no R1_MAX_ITERS cap; the local runs were capped at 1000 it/epoch), A1 vs Q8+drop50, seeds 3047 3048 3049
#   T  Traffic-32 (patchmem/make_traffic32.py, 32 random sensors), A1 vs Q8+drop50, seeds 3048 3049 (seed 3047 runs locally)
# A1 = R1, lambda_hist 0. Q8+drop50 = R1Mem, R1_MEM_QK_DIM=8, R1_MEM_SLOT_KEEP=0.5. DES tags are the runners' defaults:
#   Weather R1Wh0s<seed> / R1WMK0.5Q8c1h0s<seed>, Traffic-32 R1Th0s<seed> / R1TMK0.5Q8c1h0s<seed>.
# Jobs run JOBS_PER_GPU at a time per GPU, GPUs taken from GPUS (default: all visible).
# usage (repo root, python env active): GPUS=0,1 JOBS_PER_GPU=2 bash patchmem/server_queue.sh [job list]
#   job list default: "W:A1:3047 W:Q8:3047 W:A1:3048 W:Q8:3048 W:A1:3049 W:Q8:3049 T:A1:3048 T:Q8:3048 T:A1:3049 T:Q8:3049"
#   logs: patchmem/logs/server/<W|T>_<arm>_s<seed>.txt, queue log patchmem/logs/server/queue.txt
OUT=patchmem/logs/server; mkdir -p $OUT
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
JOBS=${1:-"W:A1:3047 W:Q8:3047 W:A1:3048 W:Q8:3048 W:A1:3049 W:Q8:3049 T:A1:3048 T:Q8:3048 T:A1:3049 T:Q8:3049"}
[ -f dataset/traffic32/traffic32.csv ] || python patchmem/make_traffic32.py
IFS=, read -ra G <<< "$GPUS"

run() {   # $1 = gpu, $2 = job D:ARM:SEED
  local gpu=$1 d arm s
  IFS=: read -r d arm s <<< "$2"
  local sh=$([ $d = W ] && echo patchmem/r1_weather.sh || echo patchmem/r1_traffic32.sh)
  local mem=$([ $arm = Q8 ] && echo "R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 R1_MEM_QK_DIM=8")
  echo "=== $2 start gpu $gpu $(date '+%F %T')" >> $OUT/queue.txt
  env CUDA_VISIBLE_DEVICES=$gpu SEED=$s $mem bash $sh > $OUT/${d}_${arm}_s$s.txt 2>&1
  echo "=== $2 done $(date '+%F %T') | test rows $(grep -c 'R1 test\]' $OUT/${d}_${arm}_s$s.txt)" >> $OUT/queue.txt
}
declare -A ON   # pid -> gpu of the running jobs
free_gpu() {     # the GPU with the fewest running jobs, if below JOBS_PER_GPU
  local g p n best= bn=$JOBS_PER_GPU
  for p in "${!ON[@]}"; do kill -0 $p 2>/dev/null || unset "ON[$p]"; done
  for g in "${G[@]}"; do
    n=0; for p in "${!ON[@]}"; do [ "${ON[$p]}" = $g ] && n=$((n + 1)); done
    [ $n -lt $bn ] && { best=$g; bn=$n; }
  done
  echo $best
}
for j in $JOBS; do
  while g=$(free_gpu); [ -z "$g" ]; do sleep 30; done
  run $g $j &
  ON[$!]=$g; sleep 30
done
wait
echo "=== all done $(date '+%F %T')" >> $OUT/queue.txt
