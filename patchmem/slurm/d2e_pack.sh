#!/bin/bash
# Usage: d2e_pack.sh <H> D:L:S [D:L:S ...]
# PAT-315: eval_decomp2 tasks (dataset:lookback:seed) at horizon H, all in parallel on one GPU; existing json skipped.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ITR_H=$1 E=patchmem/logs/server/error_path HH=$1
shift
for x in "$@"; do IFS=: read D LL S <<< "$x"; [ -f $E/${D}_H${HH}_L${LL}_decomp2err_s$S.json ] || echo "$D $LL $S"; done | xargs -P ${PAR:-2} -L 1 bash -c 'ITR_L=$1 python -u -m patchmem.eval_decomp2 --data $0 --seed $2 --arms Pd,Pd_I,Pd_err --tag decomp2err > $E/$0_H${HH}_L$1_decomp2err_s$2.txt 2>&1 || echo "FAIL decomp2err $0 $1 $2"'
