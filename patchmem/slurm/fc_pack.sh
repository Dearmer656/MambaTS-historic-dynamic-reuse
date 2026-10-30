#!/bin/bash
# Usage: fc_pack.sh <data code> <H> <L list, e.g. "96 720"> <seed list>
# PAT-315: FreshCast reproduction (eval_freshcast) for several (L, seed) tasks in parallel on one GPU.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ITR_H=$2
E=patchmem/logs/server/error_path
for LL in $3; do for S in $4; do
  ITR_L=$LL python -u -m patchmem.eval_freshcast --data $1 --seed $S > $E/$1_H$2_L${LL}_freshcast_s$S.txt 2>&1 || echo "FAIL fc $1 $2 $LL $S" &
done; done; wait
