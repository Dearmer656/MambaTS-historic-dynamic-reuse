#!/bin/bash
#SBATCH --job-name=pat307_xftest
#SBATCH --output=patchmem/logs/%j_xftest.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
P=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python
$P -u patchmem/test_r1_obs_xf.py && $P -u patchmem/test_r1_obs_vsrc_e.py && $P -u patchmem/test_r1_obs_vpath.py && $P -u patchmem/test_r1_obs_vshare.py
