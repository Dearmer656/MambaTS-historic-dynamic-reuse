#!/bin/bash
# PAT-307 offline multi-lag ICL probe (round 2). usage: sbatch patchmem/slurm/icl_lag.sbatch E|W [stride] [tag]
#SBATCH --job-name=pat307_icllag
#SBATCH --output=patchmem/logs/server/slurm_%j_icllag.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/icl_lag.py --data $1 --stride ${2:-1} --tag ${3:-v2} > patchmem/logs/server/icl_lag_${3:-v2}_$1_s3047.txt 2>&1
