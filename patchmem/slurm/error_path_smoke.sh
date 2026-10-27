#!/bin/bash
# PAT-315 smoke (review checklist): Traffic H192, phase demos, seed 3047, short training, subset eval.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=${SLURM_JOB_ID:-local}
mkdir -p patchmem/logs/server/error_path_smoke
for spec in "A two" "Bp two" "End two" "C two" "Cmean two" "C learn"; do echo $spec; done |
  xargs -P 6 -L 1 bash -c 'python -u patchmem/eval_error_path.py --variant $0 --beta $1 --demos phase --epochs 1 --iters 60 \
    --val_cases 5000 --test_cases 50000 --diag_cases 20000 --out_dir patchmem/logs/server/error_path_smoke \
    > patchmem/logs/server/error_path_smoke/T_phase_${0}_${1}_'"$J"'.txt 2>&1 || echo "FAIL $0 $1"'
