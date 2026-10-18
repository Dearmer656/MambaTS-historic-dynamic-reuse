#!/bin/bash
# PAT-306 eval-only sanity audit (patchmem/sanity_eval.py, E1..E5); waits until fewer than 2 training jobs are on the GPU.
# usage (repo root): bash patchmem/sanity_local.sh     logs: patchmem/logs/sanity/
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba_iso
while [ $(pgrep -fc "^python -u (patchmem/run_r1|run)[.]py") -ge 2 ]; do sleep 60; done
echo "=== start $(date '+%F %T')"
python -u patchmem/sanity_eval.py --parts E5,E1,E2,E3,E4 "$@"
echo "=== done $(date '+%F %T')"
