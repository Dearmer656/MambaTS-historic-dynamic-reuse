#!/bin/bash
# PAT-316 laptop task 8: where does FreshCast's gain on Electricity come from (patchmem/fc_src.py, no backbone training).
# Usage: bash laptop/run_fcsrc.sh            6 tasks (EL H192, L720 / L96 x seeds 3047, 2024, 2025), PAR at a time on the GPU
#        PAR=1 bash laptop/run_fcsrc.sh      one at a time (use this if a task runs out of GPU memory or RAM)
# Results + logs: patchmem/logs/laptop/fc_src/, committed and pushed to origin laptop after every task. Finished tasks are skipped.
set -u
cd "$(dirname "$0")/.."
export ITR_H=192 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
PAR=${PAR:-2}
OUT=patchmem/logs/laptop/fc_src
CSV=dataset/electricity/electricity.csv
mkdir -p $OUT
if [ ! -f $CSV ]; then mkdir -p dataset/electricity && gunzip -c laptop/data/electricity.csv.gz > $CSV; fi
echo "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad  $CSV" | sha256sum -c - || { echo "electricity.csv checksum FAILED"; exit 1; }
push() { ( flock 9; git add -f $OUT && git commit -q -m "laptop results: fc_src $1" && { git pull -q --rebase origin laptop; git push -q origin laptop; } ) 9>$OUT/.git.lock; }

python -m patchmem.fc_src --selftest || { echo "SELFTEST FAILED"; exit 1; }
SM=$OUT/EL_H192_L720_fcsrc_s3047_smoke.json
if [ ! -f $SM ]; then
  echo "=== smoke (EL L720 s3047, ~40 test origins) $(date)"
  ITR_L=720 python -u -m patchmem.fc_src --data EL --seed 3047 --smoke --out_dir $OUT > $OUT/smoke.txt 2>&1 \
    || { echo "SMOKE FAILED, see $OUT/smoke.txt"; tail -20 $OUT/smoke.txt; push "smoke FAILED"; exit 1; }
  tail -6 $OUT/smoke.txt; push "smoke"
fi

one() {
  local LL=$1 S=$2 J=$OUT/EL_H192_L$1_fcsrc_s$2.json LOG=$OUT/EL_H192_L$1_fcsrc_s$2.txt
  [ -f $J ] && { echo "skip EL L$LL s$S (done)"; return 0; }
  echo "commit $(git rev-parse --short HEAD)" > $LOG
  echo "=== EL L$LL s$S start $(date)"
  if ITR_L=$LL python -u -m patchmem.fc_src --data EL --seed $S --out_dir $OUT >> $LOG 2>&1; then
    echo "=== EL L$LL s$S done $(date)"; grep -E "^\[test|^\[done|^\[warn" $LOG; push "EL L$LL s$S"
  else
    echo "FAILED EL L$LL s$S, see $LOG"; tail -20 $LOG; push "EL L$LL s$S FAILED"
  fi
}
n=0
for t in "720 3047" "720 2024" "720 2025" "96 3047" "96 2024" "96 2025"; do
  one $t &
  n=$((n + 1)); [ $((n % PAR)) -eq 0 ] && wait
done
wait
echo "TASK 8 DONE"
