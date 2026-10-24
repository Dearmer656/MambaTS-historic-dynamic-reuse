#!/bin/bash
# PAT-307 alignment of the learned corrections with the residual (r.d, 2r.d - |d|^2). usage: sbatch patchmem/slurm/align_diag.sbatch E|W
#SBATCH --job-name=pat307_align
#SBATCH --output=patchmem/logs/server/slurm_%j_align.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/align_diag.py --data $1 > patchmem/logs/server/align_$1_s3047.txt 2>&1
