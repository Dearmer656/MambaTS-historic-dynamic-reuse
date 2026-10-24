#!/bin/bash
# PAT-307 task 5: joint training of TEFL + attention over the model's own past forecast errors (patchmem/tefl_att.py, R1TeflAtt).
# Arms 'pos' (learned look-back window) and 'qk' (+ low-rank Wq/Wk content selection), 3 seeds, ETTh2 then Weather; the T0r / TF0
# checkpoints in checkpoints/ are evaluated by the same code as the baseline. Order: smoke test (ETTh2, minutes), ETTh2, Weather.
# Finished (data, seed, arm) runs are kept in the result JSON and skipped on a rerun, so after an interruption just run it again.
# Logs are committed and pushed to origin laptop when everything finishes, or as soon as a step fails.
# Can run in parallel with task 4 (both push only their own log files; the push pulls first).
# usage (repo root, env active): bash laptop/run_tefl_att.sh            [E|W to run only one dataset]
cd "$(dirname "$0")/.."
OUT=patchmem/logs/laptop
mkdir -p $OUT
push() { git add -f $OUT/tefl_att_* && git commit -m "laptop results: tefl_att $1" && git pull --rebase --autostash origin laptop && git push origin laptop && echo "pushed ($1)"; }
echo "commit $(git rev-parse --short HEAD) tefl_att smoke ($(date))"
if ! python -u patchmem/tefl_att.py --data E --smoke --out $OUT > $OUT/tefl_att_smoke.txt 2>&1; then
  echo "SMOKE FAILED, see $OUT/tefl_att_smoke.txt"; tail -20 $OUT/tefl_att_smoke.txt; push "smoke FAILED"; exit 1
fi
echo "smoke ok"; grep -E "=====|test" $OUT/tefl_att_smoke.txt | tail -6 || true
for D in ${1:-E W}; do
  LOG=$OUT/tefl_att_$D.txt
  echo "tefl_att $D -> $LOG ($(date))"
  if ! python -u patchmem/tefl_att.py --data $D --out $OUT >> $LOG 2>&1; then
    echo "FAILED $D, see $LOG"; tail -20 $LOG; push "$D FAILED"; exit 1
  fi
  echo "done $D ($(date))"; grep -E "^=====" $LOG || true
  push "$D"
done
