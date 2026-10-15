#!/bin/bash
#SBATCH --job-name=pat304_r1
#SBATCH --output=patchmem/logs/%j_r1.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:55:00
# usage: sbatch --export=ALL,SEED=3047,R1_LAMBDA_HIST=0.5 patchmem/slurm/r1_etth2.sbatch
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
PY=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python bash patchmem/r1_etth2.sh
