#!/bin/bash
# PAT-315 two-granularity error model (eval_sr.py, then diag_shock_ridge.py): TASKS="E:3047 ..." (data:seed), PAR at a time; existing jsons skipped.
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
O=patchmem/logs/server/err_sr
G=patchmem/logs/server/err_sr_diag
mkdir -p $O $G
PAR=${PAR:-2}
for x in $TASKS; do
  IFS=: read d s <<< "$x"
  f=$O/${d}_H192_s${s}
  g=$G/${d}_shockridge_s${s}
  [ -f $f.json ] && [ -f $g.json ] && continue
  while [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 20; done
  { [ -f $f.json ] || python -m patchmem.eval_sr --data $d --seed $s > $f.txt 2>&1 || echo "FAIL sr $d $s"; [ -f $g.json ] || python -m patchmem.diag_shock_ridge --data $d --seed $s --chunk 16 > $g.txt 2>&1 || echo "FAIL ridge $d $s"; } &
done
wait
echo done
