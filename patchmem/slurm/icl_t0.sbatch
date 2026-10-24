#!/bin/bash
# PAT-307 offline in-window ICL on top of T0r / TF0. usage: sbatch patchmem/slurm/icl_t0.sbatch E|W [stride]
#SBATCH --job-name=pat307_icl_t0
#SBATCH --output=patchmem/logs/server/slurm_%j_icl_t0.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/icl_t0.py --data $1 --stride ${2:-1} > patchmem/logs/server/icl_t0_$1.txt 2>&1
