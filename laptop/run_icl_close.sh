#!/bin/bash
# PAT-307 icl_close: closing probe for the TEFL-attention line (bias memory, H->H fixed key, learned low-rank Wq/Wk), frozen T0r /
# TF0 checkpoints in checkpoints/ (seeds 3047-3049). Order: a short crash test (smoke) on ETTh2, then ETTh2 (stride 1) and Weather
# (stride 4, train queries subsampled x4). When everything finishes, or as soon as a step fails, the logs are committed and pushed
# to origin laptop. usage (repo root, env active): bash laptop/run_icl_close.sh            [E|W to run only one dataset]
cd "$(dirname "$0")/.."
OUT=patchmem/logs/laptop
mkdir -p $OUT
push() { git add -f $OUT/icl_close_* && git commit -m "laptop results: icl_close $1" && git push origin laptop && echo "pushed ($1)"; }
echo "commit $(git rev-parse --short HEAD) icl_close smoke ($(date))"
if ! python -u patchmem/icl_close.py --data E --smoke --out $OUT > $OUT/icl_close_smoke.txt 2>&1; then
  echo "SMOKE FAILED, see $OUT/icl_close_smoke.txt"; tail -20 $OUT/icl_close_smoke.txt; push "smoke FAILED"; exit 1
fi
echo "smoke ok"; grep -E "criterion" $OUT/icl_close_smoke.txt || true
for D in ${1:-E W}; do
  case $D in E) st=1; sub=1 ;; W) st=4; sub=4 ;; *) echo "unknown data $D (E = ETTh2, W = Weather)"; exit 1 ;; esac
  LOG=$OUT/icl_close_$D.txt
  echo "icl_close $D stride $st tr_sub $sub -> $LOG ($(date))"
  if ! python -u patchmem/icl_close.py --data $D --stride $st --tr_sub $sub --out $OUT > $LOG 2>&1; then
    echo "FAILED $D, see $LOG"; tail -20 $LOG; push "$D FAILED"; exit 1
  fi
  echo "done $D ($(date))"; grep -E "criterion" $LOG || true
done
push "${1:-E W}"
