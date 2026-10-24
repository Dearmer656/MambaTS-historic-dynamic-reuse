#!/bin/bash
# PAT-307 residual covariance (future vs history residuals, A1 and A0, train/val/test). usage: sbatch patchmem/slurm/resid_cov.sbatch E|W
#SBATCH --job-name=pat307_rcov
#SBATCH --output=patchmem/logs/server/slurm_%j_rcov.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/resid_cov.py --data $1 > patchmem/logs/server/resid_cov_$1_s3047.txt 2>&1
