#!/bin/bash
# PAT-316 laptop task 9 (PAT-319): FreshCast scorer conditioned on f / RoPE age / PaTH path (patchmem/fc_cond.py).
# Usage: bash laptop/run_fccond.sh           3 tasks (EL H192 L720 x seeds 3047, 2024, 2025), one at a time
#        PAR=2 bash laptop/run_fccond.sh     two at a time (only if the smoke peak GPU memory is below ~7 GB)
# Results + logs: patchmem/logs/laptop/fc_cond/, committed and pushed to origin laptop after every task. Finished tasks are skipped.
set -u
cd "$(dirname "$0")/.."
export ITR_H=192 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
PAR=${PAR:-1}
OUT=patchmem/logs/laptop/fc_cond
CSV=dataset/electricity/electricity.csv
mkdir -p $OUT
if [ ! -f $CSV ]; then mkdir -p dataset/electricity && gunzip -c laptop/data/electricity.csv.gz > $CSV; fi
echo "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad  $CSV" | sha256sum -c - || { echo "electricity.csv checksum FAILED"; exit 1; }
push() { ( flock 9; git add -f $OUT && git commit -q -m "laptop results: fc_cond $1" && { git pull -q --rebase origin laptop; git push -q origin laptop; } ) 9>$OUT/.git.lock; }

python -m patchmem.fc_cond --selftest || { echo "SELFTEST FAILED"; exit 1; }
SM=$OUT/EL_H192_L720_fccond_s3047_smoke.json
if [ ! -f $SM ]; then
  echo "=== smoke (EL L720 s3047, ~40 test origins) $(date)"
  ITR_L=720 python -u -m patchmem.fc_cond --data EL --seed 3047 --smoke --out_dir $OUT > $OUT/smoke.txt 2>&1 \
    || { echo "SMOKE FAILED, see $OUT/smoke.txt"; tail -20 $OUT/smoke.txt; push "smoke FAILED"; exit 1; }
  tail -6 $OUT/smoke.txt; push "smoke"
fi

one() {
  local LL=$1 S=$2 J=$OUT/EL_H192_L$1_fccond_s$2.json LOG=$OUT/EL_H192_L$1_fccond_s$2.txt
  [ -f $J ] && { echo "skip EL L$LL s$S (done)"; return 0; }
  echo "commit $(git rev-parse --short HEAD)" > $LOG
  echo "=== EL L$LL s$S start $(date)"
  if ITR_L=$LL python -u -m patchmem.fc_cond --data EL --seed $S --out_dir $OUT >> $LOG 2>&1; then
    echo "=== EL L$LL s$S done $(date)"; grep -E "^\[test|^\[done|^\[warn" $LOG; push "EL L$LL s$S"
  else
    echo "FAILED EL L$LL s$S, see $LOG"; tail -20 $LOG; push "EL L$LL s$S FAILED"
  fi
}
n=0
for t in "720 3047" "720 2024" "720 2025"; do
  one $t &
  n=$((n + 1)); [ $((n % PAR)) -eq 0 ] && wait
done
wait
echo "TASK 9 DONE"
