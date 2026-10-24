#!/bin/bash
# PAT-307 offline ICL experts + cross-window lag memory (hedge). usage: sbatch patchmem/slurm/icl_hedge.sbatch E|W [stride]
#SBATCH --job-name=pat307_hedge
#SBATCH --output=patchmem/logs/server/slurm_%j_hedge.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/icl_hedge.py --data $1 --stride ${2:-1} > patchmem/logs/server/icl_hedge_$1_s3047.txt 2>&1
