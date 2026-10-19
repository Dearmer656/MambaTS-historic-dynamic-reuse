#!/bin/bash
#SBATCH --job-name=pat307_lead
#SBATCH --output=patchmem/logs/%j_lead.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# usage: sbatch patchmem/slurm/lead_fb.sbatch E|W SEED [GRID]
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
python -u patchmem/lead_fb_eval.py --data $1 --seed $2 ${3:+--grid $3}
