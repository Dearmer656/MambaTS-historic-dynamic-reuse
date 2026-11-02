#!/bin/bash
# PAT-315 Traffic long-lookback dev runs (docker, one A6000): eval_sr v3 on the frozen iTransformer L720, H192 / H336, 3 seeds.
# v3 = shock lags + fixed rotary base, theta frozen, BPTT one week, early stopping on val; arms s1_R, s1_D, s1_P, s1_P2.
# Config choices are made on VAL only; test numbers are written but not used for selection.
# Usage (repo root):  bash docker/run_traffic_dev.sh            all 6 runs, one after the other (finished runs skipped)
#                     bash docker/run_traffic_dev.sh smoke      only the smoke test
# Results + logs: patchmem/logs/docker/sr3/, committed and pushed to origin docker after every run.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python3}
OUT=patchmem/logs/docker/sr3
CSV=dataset/traffic/traffic.csv
mkdir -p $OUT dataset/traffic
if [ ! -f $CSV ] || [ -L $CSV ]; then rm -f $CSV; gunzip -c docker/data/traffic.csv.gz > $CSV; fi
echo "cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16  $CSV" | sha256sum -c - || { echo "traffic.csv checksum FAILED"; exit 1; }
V3="--base shrot --freeze_theta --bptt_h 168 --early_stop --arms s1_R,s1_D,s1_P,s1_P2 --fc_dir docker/refs"
push() { git add -f $OUT && git commit -q -m "docker results: sr3 $1" && { git pull -q --rebase origin docker; git push -q origin docker; }; }

echo "=== smoke $(date)"
if ! ITR_L=720 ITR_H=192 $PY -u -m patchmem.eval_sr --data T --seed 3047 --epochs 1 --max_c 4 $V3 --out_dir ${OUT}_smoke > $OUT/smoke.txt 2>&1; then
  echo "SMOKE FAILED"; tail -30 $OUT/smoke.txt; push "smoke FAILED"; exit 1
fi
tail -6 $OUT/smoke.txt; echo "SMOKE OK"
[ "${1:-}" = smoke ] && exit 0

for t in "192 3047" "336 3047" "192 2024" "336 2024" "192 2025" "336 2025"; do
  set -- $t; HH=$1; S=$2; J=$OUT/T_H${HH}_L720_s$S.json; LOG=$OUT/T_H${HH}_L720_s$S.txt
  [ -f $J ] && { echo "skip T H$HH s$S (done)"; continue; }
  echo "commit $(git rev-parse --short HEAD)" > $LOG
  echo "=== T L720 H$HH s$S  $(date)"
  if ITR_L=720 ITR_H=$HH $PY -u -m patchmem.eval_sr --data T --seed $S $V3 --out_dir $OUT >> $LOG 2>&1; then
    push "T H$HH s$S"; grep "\[test\]" $LOG | tail -14
  else
    echo "FAILED T H$HH s$S, see $LOG"; tail -20 $LOG; push "T H$HH s$S FAILED"; exit 1
  fi
done
echo "TRAFFIC DEV DONE"
