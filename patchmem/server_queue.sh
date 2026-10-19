#!/bin/bash
# PAT-304 / PAT-306 server queue: the runs the laptop cannot afford. Same recipe for every arm (no tuning of either arm).
# A job is DATASET:ARM:SEED.
#   DATASET  E = ETTh2 (patchmem/r1_etth2.sh), W = Weather FULL budget (patchmem/r1_weather.sh, no R1_MAX_ITERS cap; the local runs were
#            capped at 1000 it/epoch), T = Traffic-32 (patchmem/r1_traffic32.sh, 32 random sensors, patchmem/make_traffic32.py)
#   ARM      A1  R1, lambda_hist 0
#            Q8  Q8+drop50: R1Mem, R1_MEM_QK_DIM=8, R1_MEM_SLOT_KEEP=0.5
#            U   uniform+drop50: R1Mem with selection-free (uniform) attention trained from scratch, R1_MEM_UNIFORM=1, R1_MEM_SLOT_KEEP=0.5
#            O   observation-space error memory (patchmem/r1_obsmem.py): values = the backbone's lookback errors, the retrieved correction
#                is added to the forecast; same retrieval recipe as Q8+drop50 (R1_OBS=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5), DES ...ObK0.5Q8...
#            PAT-306 controls (ETTh2 only, same as patchmem/sanity_train.sh):
#            T2  A1 with d_model 26 (315,873 params ~ Q8 315,400)                 -> gain from capacity?
#            T3  Q8+drop50 with CorrNet residuals rolled across the batch (R1_CORR_SHUFFLE=1) -> gain from the sample's own information?
#            T4a A1, fixed L=720 / H=96 (train and val)  T4b Q8+drop50, same     -> gain only in the one-model-for-all-L/H setting?
# DES tags are the runners' defaults (E: R1h0s<seed>, R1MK0.5Q8c1h0s<seed>, R1MUK0.5c1h0s<seed>; W: R1W...; T: R1T...), the controls
# use the names of patchmem/sanity_train.sh (R1h0dm26s, R1MSK0.5Q8c1h0s, R1h0F720x96s, R1MK0.5Q8c1h0F720x96s).
# Jobs run JOBS_PER_GPU at a time per GPU, GPUs taken from GPUS (default: all visible); each job goes to the least-loaded GPU.
# usage (repo root, python env active): GPUS=0,1 JOBS_PER_GPU=2 bash patchmem/server_queue.sh [job list]
#   e.g. the observation-space memory on all datasets: bash patchmem/server_queue.sh "E:O:3047 E:O:3048 E:O:3049 T:O:3047 T:O:3048 T:O:3049 W:O:3047 W:O:3048 W:O:3049"
#   two queues running at once do not see each other's jobs: give them disjoint GPUS.
#   logs: patchmem/logs/server/<D>_<arm>_s<seed>.txt, queue log patchmem/logs/server/queue.txt
OUT=patchmem/logs/server; mkdir -p $OUT
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
# default, in priority order: PAT-306 controls + uniform seeds on ETTh2 (cheap), Traffic-32 seeds and uniform, Weather full x3 seeds
JOBS=${1:-"E:T2:3047 E:T3:3047 E:U:3048 E:U:3049 E:T4a:3047 E:T4b:3047
T:U:3047 T:A1:3048 T:Q8:3048 T:U:3048 T:A1:3049 T:Q8:3049 T:U:3049
W:A1:3047 W:Q8:3047 W:A1:3048 W:Q8:3048 W:A1:3049 W:Q8:3049"}
[ -f dataset/traffic32/traffic32.csv ] || python patchmem/make_traffic32.py
IFS=, read -ra G <<< "$GPUS"

run() {   # $1 = gpu, $2 = job D:ARM:SEED
  local gpu=$1 d arm s sh env=
  IFS=: read -r d arm s <<< "$2"
  case $d in E) sh=patchmem/r1_etth2.sh ;; W) sh=patchmem/r1_weather.sh ;; T) sh=patchmem/r1_traffic32.sh ;; esac
  local Q8="R1_MEM=1 R1_MEM_SLOT_KEEP=0.5 R1_MEM_QK_DIM=8" FIX="R1_LMIN=720 R1_LMAX=720 R1_HMIN=2 R1_HMAX=2 R1_VAL_GRID=720:96"
  case $arm in
    A1)  env= ;;
    Q8)  env=$Q8 ;;
    U)   env="R1_MEM=1 R1_MEM_UNIFORM=1 R1_MEM_SLOT_KEEP=0.5" ;;
    O)   env="R1_OBS=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5" ;;
    T2)  env="EXTRA=--d_model=26 DES=R1h0dm26s$s" ;;
    T3)  env="$Q8 R1_CORR_SHUFFLE=1" ;;
    T4a) env="$FIX DES=R1h0F720x96s$s" ;;
    T4b) env="$FIX $Q8 DES=R1MK0.5Q8c1h0F720x96s$s" ;;
    *)   echo "=== $2 unknown arm" >> $OUT/queue.txt; return ;;
  esac
  echo "=== $2 start gpu $gpu $(date '+%F %T')" >> $OUT/queue.txt
  env CUDA_VISIBLE_DEVICES=$gpu SEED=$s $env bash $sh > $OUT/${d}_${arm}_s$s.txt 2>&1
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
