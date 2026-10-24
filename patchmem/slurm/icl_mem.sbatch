#!/bin/bash
# PAT-307 offline memory of own past forecasts (TEFL dual attention). usage: sbatch patchmem/slurm/icl_mem.sbatch E|W [stride]
#SBATCH --job-name=pat307_icl_mem
#SBATCH --output=patchmem/logs/server/slurm_%j_icl_mem.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/icl_mem.py --data $1 --stride ${2:-1} > patchmem/logs/server/icl_mem_$1.txt 2>&1
