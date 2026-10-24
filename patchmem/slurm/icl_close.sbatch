#!/bin/bash
# PAT-307 offline closing probe for the TEFL-attention line (bias memory, H->H key, learned low-rank Wq/Wk). usage: sbatch patchmem/slurm/icl_close.sbatch E|W [stride]
#SBATCH --job-name=pat307_icl_close
#SBATCH --output=patchmem/logs/server/slurm_%j_icl_close.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/icl_close.py --data $1 --stride ${2:-1} --tr_sub ${3:-1} > patchmem/logs/server/icl_close_$1.txt 2>&1
