#!/bin/bash
# PAT-307 ErrPath-shared (error path in the P-dim patch space, same Householder for every variable), variable-length R1.
# Identical settings to the cluster job patchmem/slurm/vshare_job.sbatch. usage (repo root, env active): bash laptop/run_errpath_shared.sh E|W SEED
set -e
cd "$(dirname "$0")/.."
D=$1; s=$2
[ -n "$D" ] && [ -n "$s" ] || { echo "usage: bash laptop/run_errpath_shared.sh E|W SEED"; exit 1; }
export SEED=$s R1_OBS=1 R1_OBS_VPATH=1 R1_OBS_VSHARE=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5
case $D in
  E) export DES=R1ObK0.5Q8PVsh0s$s; S=patchmem/r1_etth2.sh ;;
  W) export DES=R1WObK0.5Q8PVsh0s$s; S=patchmem/r1_weather.sh ;;
  *) echo "unknown data $D (E = ETTh2, W = Weather)"; exit 1 ;;
esac
mkdir -p patchmem/logs/laptop
LOG=patchmem/logs/laptop/${D}_OpathVs_s$s.txt
echo "commit $(git rev-parse --short HEAD) ErrPath-shared $D seed $s DES $DES -> $LOG"
bash $S > $LOG 2>&1
echo "done: $LOG"; grep "\[R1Mem audit test\]" $LOG | cut -c1-75
