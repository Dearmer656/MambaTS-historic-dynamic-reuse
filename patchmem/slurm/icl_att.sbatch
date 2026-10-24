#!/bin/bash
# PAT-307 offline lag-specific softmax attention + fusion. usage: sbatch patchmem/slurm/icl_att.sbatch E|W [stride]
#SBATCH --job-name=pat307_att
#SBATCH --output=patchmem/logs/server/slurm_%j_att.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/icl_att.py --data $1 --stride ${2:-1} > patchmem/logs/server/icl_att_$1.txt 2>&1
