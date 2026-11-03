#!/bin/bash
# PAT-315 Step 2 on one elm72 GPU: 3 seeds in parallel on the same card, each H192 then H336.
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ITR_L=720
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
V='--base shrot --freeze_theta --bptt_h 168 --early_stop --val_split --arms s1_P,s1_GF,s1_Pfix,s1_Pdiag --mech s1_P,s1_Pfix,s1_Pdiag --dump --out_dir patchmem/logs/server/err_s2'
mkdir -p patchmem/logs/server/err_s2
for s in 3047 2024 2025; do
  { for hh in 192 336; do
      [ -f patchmem/logs/server/err_s2/T_H${hh}_L720_s$s.json ] && continue
      ITR_H=$hh python -m patchmem.eval_sr --data T --seed $s $V > patchmem/logs/server/err_s2/T_H${hh}_L720_s$s.txt 2>&1 || echo "FAIL s2 H$hh s$s"
    done; } &
done
wait
echo done
