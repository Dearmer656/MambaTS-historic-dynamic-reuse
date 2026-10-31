#!/bin/bash
# PAT-315 lead structure of the forecast error (diag_err_lead.py): TASKS="E:3047:192 ..." (data:seed:H), PAR at a time.
# T / EL keep 5664 + 2H origins before n_tr (as errval_pack.sh); existing jsons skipped.
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
O=patchmem/logs/server/err_lead
mkdir -p $O
PAR=${PAR:-2}
for x in $TASKS; do
  IFS=: read d s h <<< "$x"
  f=$O/${d}_H${h}_s${s}
  [ -f $f.json ] && continue
  X=""; { [ $d = T ] || [ $d = EL ]; } && X="--m_tr $((5664 + 2 * h)) --chunk 8"
  while [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 20; done
  ITR_H=$h python -m patchmem.diag_err_lead --data $d --seed $s $X > $f.txt 2>&1 || echo "FAIL $d $s $h" &
done
wait
echo done
