#!/bin/bash
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=${SLURM_JOB_ID:-local}
for D in T W; do for S in 3047 2024 2025; do python -u patchmem/diag_itransformer_tefl_twolayer_w1.py --data $D --seed $S > patchmem/logs/server/itr_twolayer_diag/${D}_s${S}_${J}.txt 2>&1 & done; done
wait
