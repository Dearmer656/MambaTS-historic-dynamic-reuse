#!/bin/bash
# PAT-315 carry LR diagnosis: s1_P, Traffic L720 H192 seed 3047, Step 2 settings, 6 lr configs on one card.
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ITR_L=720 ITR_H=192
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
V='--data T --seed 3047 --base shrot --freeze_theta --bptt_h 168 --early_stop --val_split --arms s1_P --log_updates'
O=patchmem/logs/server/err_lr
for cfg in "c3e-3:--lr 3e-3" "c1e-3:--lr 1e-3" "c3e-4:--lr 3e-4" "c1e-4:--lr 1e-4" "wc3e-3:--lr 3e-3 --lr_sched cos --warmup 500" "wc1e-3:--lr 1e-3 --lr_sched cos --warmup 500"; do
  n=${cfg%%:*}; mkdir -p $O/$n
  python -m patchmem.eval_sr $V ${cfg#*:} --out_dir $O/$n > $O/$n/log.txt 2>&1 || echo "FAIL $n" &
done
wait
echo done
