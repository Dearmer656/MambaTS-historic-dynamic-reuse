#!/bin/bash
# PAT-315 eval_sr v2 (--base shrot --freeze_theta --bptt_h 168): TASKS="E:3047 ..." (data:seed), PAR at a time; existing jsons skipped.
# SMOKE=1 first runs a 1-epoch, 2-channel ETTh2 smoke and stops on failure.
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
O=patchmem/logs/server/err_sr2
V2="--base shrot --freeze_theta --bptt_h 168"
mkdir -p $O
if [ -n "$SMOKE" ]; then
  python -m patchmem.eval_sr --data E --seed 3047 --epochs 1 --max_c 2 $V2 --out_dir ${O}_smoke || { echo "FAIL smoke"; exit 1; }
fi
PAR=${PAR:-2}
for x in $TASKS; do
  IFS=: read d s <<< "$x"
  f=$O/${d}_H192_s${s}
  [ -f $f.json ] && continue
  while [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 20; done
  { python -m patchmem.eval_sr --data $d --seed $s $V2 --out_dir $O > $f.txt 2>&1 || echo "FAIL sr2 $d $s"; } &
done
wait
echo done
