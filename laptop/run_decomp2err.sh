#!/bin/bash
# PAT-315 laptop task 7: eval_decomp2 Pd_err (PaTH transition inputs = observed errors + d mu + d log sd), Electricity, H192.
# Usage: bash laptop/run_decomp2err.sh            (all 4 tasks, one after the other; finished tasks are skipped)
# Results + logs: patchmem/logs/laptop/decomp2err/, committed and pushed to origin laptop after every task.
set -u
cd "$(dirname "$0")/.."
export ITR_H=192 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=patchmem/logs/laptop/decomp2err
CSV=dataset/electricity/electricity.csv
if [ ! -f $CSV ]; then mkdir -p dataset/electricity && gunzip -c laptop/data/electricity.csv.gz > $CSV; fi
echo "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad  $CSV" | sha256sum -c - || { echo "electricity.csv checksum FAILED"; exit 1; }
push() { git add -f $OUT && git commit -q -m "laptop results: decomp2err $1" && { git pull -q --rebase origin laptop; git push -q origin laptop; }; }
for t in "720 2025" "96 2025" "720 2024" "96 2024"; do
  set -- $t; LL=$1; S=$2; J=$OUT/EL_H192_L${LL}_decomp2err_s$S.json; LOG=$OUT/EL_H192_L${LL}_decomp2err_s$S.txt
  [ -f $J ] && { echo "skip EL L$LL s$S (done)"; continue; }
  echo "commit $(git rev-parse --short HEAD)" > $LOG
  echo "=== EL L$LL s$S  $(date)"
  if ITR_L=$LL python -u -m patchmem.eval_decomp2 --data EL --seed $S --arms Pd,Pd_I,Pd_err --tag decomp2err --out_dir $OUT >> $LOG 2>&1; then
    push "EL L$LL s$S"; tail -12 $LOG
  else
    echo "FAILED EL L$LL s$S, see $LOG"; tail -20 $LOG; push "EL L$LL s$S FAILED"; exit 1
  fi
done
echo "TASK 7 DONE"
