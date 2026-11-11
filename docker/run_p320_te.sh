#!/bin/bash
# PAT-320 task-embedding control (docker, one A6000): one Timer-XL trained jointly on 7 LTSF datasets (channel independent)
# + a learned per-dataset embedding added to every input token (zero init), then the error-carry heads c (nopath) and
# d (path) on that frozen backbone. Same recipe / splits / criteria as the cluster runs without the embedding.
# Usage (repo root):  bash docker/run_p320_te.sh          smoke, then 3 seeds x (stage 1, carry path, carry nopath)
#                     bash docker/run_p320_te.sh smoke    only the smoke test
# Finished runs are skipped (re-run the same command after an interruption).
# Results + logs: openltm/pat_logs/docker_te/, committed and pushed to origin docker after every run.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python3}
R=openltm; OUT=$R/pat_logs/docker_te; D=$R/dataset
mkdir -p $OUT/smoke $D/ETT-small $D/weather $D/electricity $D/traffic
unpack() { [ -f $D/$2 ] && [ ! -L $D/$2 ] || { rm -f $D/$2; gunzip -c $1 > $D/$2; }; }
for f in ETTh1 ETTh2 ETTm1 ETTm2; do unpack docker/data/p320/$f.csv.gz ETT-small/$f.csv; done
unpack docker/data/p320/weather.csv.gz weather/weather.csv
unpack docker/data/p320/electricity.csv.gz electricity/electricity.csv
unpack docker/data/traffic.csv.gz traffic/traffic.csv
(cd $D && sha256sum -c ../../docker/data/p320/SHA256SUMS) || { echo "dataset checksum FAILED"; exit 1; }
push() { git add -f $OUT && git commit -q -m "docker results: p320 te $1" && { git pull -q --rebase origin docker; git push -q origin docker; }; }

echo "=== smoke $(date)"
if ! { $PY $R/pat320/test_task_emb.py && $PY -u $R/pat320/joint.py --smoke --task_emb --seed 2021 --out $OUT/smoke \
       && $PY -u $R/pat320/carry.py --smoke --ckpt $OUT/smoke/joint_te_s2021_smoke.pth --arm path --data ETTh1,Traffic --out $OUT/smoke; } \
       > $OUT/smoke.txt 2>&1; then
  echo "SMOKE FAILED"; tail -30 $OUT/smoke.txt; push "smoke FAILED"; exit 1
fi
grep -E "no-op|grad rows|reload|\[test\]" $OUT/smoke.txt | tail -12; rm -f $OUT/smoke/*.pth; echo "SMOKE OK"
[ "${1:-}" = smoke ] && exit 0

for S in 2021 2022 2023; do
  H="commit $(git rev-parse --short HEAD) seed $S host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  CK=$OUT/joint_te_s$S.pth
  if [ -f $OUT/joint_te_s$S.json ]; then echo "skip stage 1 s$S (done)"; else
    echo "=== stage 1 te s$S  $(date)"; echo $H > $OUT/joint_te_s$S.txt
    if $PY -u $R/pat320/joint.py --task_emb --seed $S --out $OUT >> $OUT/joint_te_s$S.txt 2>&1; then
      push "stage 1 s$S"; grep "\[test\]" $OUT/joint_te_s$S.txt
    else echo "FAILED stage 1 s$S"; tail -20 $OUT/joint_te_s$S.txt; push "stage 1 s$S FAILED"; exit 1; fi
  fi
  for A in path nopath; do
    J=$OUT/carry_${A}_K14_s${S}_joint_te_s$S.json; LOG=$OUT/carry_${A}_te_s$S.txt
    [ -f $J ] && { echo "skip carry $A s$S (done)"; continue; }
    echo "=== carry $A te s$S  $(date)"; echo $H > $LOG
    if $PY -u $R/pat320/carry.py --ckpt $CK --arm $A --seed $S --out $OUT >> $LOG 2>&1; then
      push "carry $A s$S"; grep "\[test\]" $LOG
    else echo "FAILED carry $A s$S"; tail -20 $LOG; push "carry $A s$S FAILED"; exit 1; fi
  done
done
echo "P320 TE DONE"
