#!/bin/bash
# Usage: obs_train_pack.sh <data code> <L> <H>
# PAT-315: train the iTransformer base at (L, H) for seeds 3047/2024/2025 in parallel on one GPU, then run the
# observation-to-continuation ICL eval (eval_obs_icl, pools phase,tiled) for the 3 seeds in parallel.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ITR_L=$2 ITR_H=$3
D=patchmem/logs/server; E=$D/error_path
for S in 3047 2024 2025; do
  if [ ! -f checkpoints/iTransformer_base_$1_L$2_H$3_s$S/checkpoint.pth ]; then
    python -u -m patchmem.run_itransformer_base --data $1 --seed $S --seq_len $2 --pred_len $3 > $D/itr_base_$1_L$2_H$3_s$S.txt 2>&1 || echo "FAIL train $1 $2 $3 $S" &
  fi
done; wait
for S in 3047 2024 2025; do
  python -u -m patchmem.eval_obs_icl --data $1 --seed $S --pools phase,tiled > $E/$1_H$3_L$2_obsicl2_s$S.txt 2>&1 || echo "FAIL obs $1 $2 $3 $S" &
done; wait
