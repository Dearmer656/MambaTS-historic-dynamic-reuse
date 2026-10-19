#!/bin/bash
# A1 + naive observation-error feedback next to the trained arms (eval only). usage: sbatch patchmem/slurm/naive_fb.sbatch E|W SEED ARMS [GRID]
#SBATCH --job-name=pat304_naive
#SBATCH --output=patchmem/logs/naive/slurm_%j.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
python -u patchmem/naive_fb_eval.py --data $1 --seed $2 --arms "$3" ${4:+--grid $4} ${5:+--out $5}
