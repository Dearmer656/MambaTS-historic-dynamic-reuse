#!/bin/bash
# PAT-315 eval_sr v3 on the server 3090s: Traffic L720, TASKS="192:3047 336:3047 ..." (H:seed), one after the other; existing jsons skipped.
# arms: s1_P (same-machine anchor), s1_G, s1_GF (selective rotary), s1_Ponly. SMOKE=1 first runs a 1-epoch 4-channel smoke.
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
O=patchmem/logs/server/err_sr3
V3="--base shrot --freeze_theta --bptt_h 168 --early_stop --arms ${ARMS:-s1_P,s1_G,s1_GF,s1_Ponly}"
mkdir -p $O
if [ -n "$SMOKE" ]; then
  ITR_L=720 ITR_H=192 python -m patchmem.eval_sr --data T --seed 3047 --epochs 1 --max_c 4 $V3 --out_dir ${O}_smoke || { echo "FAIL smoke"; exit 1; }
fi
for x in $TASKS; do
  IFS=: read hh s <<< "$x"
  f=$O/T_H${hh}_L720_s${s}
  [ -f $f.json ] && continue
  ITR_L=720 ITR_H=$hh python -m patchmem.eval_sr --data T --seed $s $V3 --out_dir $O > $f.txt 2>&1 || echo "FAIL sr3 H$hh s$s"
done
echo done
