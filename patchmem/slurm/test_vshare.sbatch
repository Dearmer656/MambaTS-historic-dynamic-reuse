#!/bin/bash
#SBATCH --job-name=pat304_obstest
#SBATCH --output=patchmem/logs/%j_obstest.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/test_r1_obs_vsrc_e.py && /cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/test_r1_obs_vshare.py && /cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/test_r1_obs_vpath.py
