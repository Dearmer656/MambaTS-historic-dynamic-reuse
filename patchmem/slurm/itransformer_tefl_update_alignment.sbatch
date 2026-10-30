#!/bin/bash
#SBATCH --job-name=itr_tefl_align
#SBATCH --output=patchmem/logs/server/itr_tefl_align_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m patchmem.diag_tefl_update_alignment
