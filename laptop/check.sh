#!/bin/bash
# Environment check (~15-30 min; the smoke run trains 100 iterations, then evaluates the full test grid): GPU kernels + the ErrPath / ErrPath-shared unit tests + a 100-iteration training smoke run.
# usage (repo root, env active): bash laptop/check.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p patchmem/logs/laptop
python -u patchmem/test_r1_obs_vshare.py
python -u patchmem/test_r1_obs_vpath.py
SEED=3047 R1_OBS=1 R1_OBS_VPATH=1 R1_OBS_VSHARE=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5 EPOCHS=1 R1_MAX_ITERS=100 R1_VAL_STRIDE=64 \
  DES=laptopsmoke bash patchmem/r1_etth2.sh 2>&1 | tee patchmem/logs/laptop/smoke.txt | grep -E --line-buffered "R1Obs:|Epoch: 1|audit test\]|Error" || true
grep -q "audit test\]" patchmem/logs/laptop/smoke.txt && echo "CHECK OK" || { echo "CHECK FAILED, see patchmem/logs/laptop/smoke.txt"; exit 1; }
