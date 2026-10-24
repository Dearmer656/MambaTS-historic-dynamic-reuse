#!/bin/bash
# PAT-307 icl_mem (TEFL-dual attention over the model's own past forecasts), offline probe on the frozen T0r (ETTh2) / TF0 (Weather)
# checkpoints committed in checkpoints/ (seeds 3047-3049). Same settings as the cluster job patchmem/slurm/icl_mem.sbatch
# (ETTh2 stride 1, Weather stride 4). When both finish (or fail), the logs are committed and pushed to origin laptop automatically.
# usage (repo root, env active): bash laptop/run_icl_mem.sh            [E|W to run only one]
set -e
cd "$(dirname "$0")/.."
mkdir -p patchmem/logs/server patchmem/logs/laptop
DS=${1:-"E W"}
for D in $DS; do
  case $D in E) st=1 ;; W) st=4 ;; *) echo "unknown data $D (E = ETTh2, W = Weather)"; exit 1 ;; esac
  LOG=patchmem/logs/laptop/icl_mem_$D.txt
  echo "commit $(git rev-parse --short HEAD) icl_mem $D stride $st -> $LOG ($(date))"
  python -u patchmem/icl_mem.py --data $D --stride $st > $LOG 2>&1 || echo "FAILED $D (see $LOG; pushed anyway)"
  cp patchmem/logs/server/icl_mem_$D.json patchmem/logs/laptop/ 2>/dev/null || true
  echo "done $D ($(date))"; grep -E "best att|criterion" $LOG || true
done
git add -f patchmem/logs/laptop/icl_mem_*
git commit -m "laptop results: icl_mem $DS"
git push origin laptop
echo "pushed"
