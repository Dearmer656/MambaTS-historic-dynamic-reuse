#!/bin/bash
# PAT-315 PaTH retrieval (eval_pathret): usage [FIT=val_fit EP=12 OSUF=_vf] pathret.sh "<arms in order>"; 3 seeds in parallel on one GPU, arms sequential.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ITR_H=192 ITR_L=720
FIT=${FIT:-train}; EP=${EP:-8}; O=patchmem/logs/server/err_pathret${OSUF}
mkdir -p $O
for S in 3047 2024 2025; do
  { for A in $1; do
      [ -f $O/$A/T_H192_L720_s$S.json ] && continue
      python -u -m patchmem.eval_pathret --seed $S --arm $A --epochs $EP --steps 2000 --log_every 250 --fit $FIT --out_dir $O > $O/${A}_s$S.txt 2>&1 || echo "FAIL $A $S"
    done; } &
done; wait
