#!/bin/bash
# PAT-307 offline in-context regression probe on A1/A0 residuals. usage: sbatch patchmem/slurm/icl_probe.sbatch E|W
#SBATCH --job-name=pat307_icl
#SBATCH --output=patchmem/logs/server/slurm_%j_icl.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/icl_probe.py --data $1 > patchmem/logs/server/icl_probe_$1_s3047.txt 2>&1
