#!/bin/bash
# PAT-315 smoke: one seed, tiny training, subset eval; checks speed/memory per variant.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
J=${SLURM_JOB_ID:-local}
mkdir -p patchmem/logs/server/error_path_smoke
for V in A C Crev; do
  python -u patchmem/eval_error_path.py --variant $V --demos dense --epochs 1 --iters 30 --val_cases 2000 --test_cases 5000 \
    --out_dir patchmem/logs/server/error_path_smoke > patchmem/logs/server/error_path_smoke/T_dense_${V}_${J}.txt 2>&1 &
done
wait
