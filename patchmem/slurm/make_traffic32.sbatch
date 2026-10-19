#!/bin/bash
#SBATCH --job-name=pat304_t32data
#SBATCH --output=patchmem/logs/%j_t32data.txt
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/make_traffic32.py
