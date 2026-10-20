#!/bin/bash
#SBATCH --job-name=pat307_otopk
#SBATCH --output=patchmem/logs/%j_obs_topk.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# usage: sbatch patchmem/slurm/obs_topk.sbatch E|W SEED [--grid ..] [--p24]
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
python -u patchmem/obs_topk_eval.py --data $1 --seed $2 ${@:3}
