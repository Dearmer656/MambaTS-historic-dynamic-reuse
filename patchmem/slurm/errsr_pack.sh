#!/bin/bash
# PAT-315 two-granularity error model (eval_sr.py): TASKS="E:3047 ..." (data:seed), PAR at a time; existing jsons skipped.
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
O=patchmem/logs/server/err_sr
mkdir -p $O
PAR=${PAR:-2}
for x in $TASKS; do
  IFS=: read d s <<< "$x"
  f=$O/${d}_H192_s${s}
  [ -f $f.json ] && continue
  while [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 20; done
  python -m patchmem.eval_sr --data $d --seed $s > $f.txt 2>&1 || echo "FAIL $d $s" &
done
wait
echo done
