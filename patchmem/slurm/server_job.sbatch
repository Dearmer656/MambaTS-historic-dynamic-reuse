#!/bin/bash
# one PAT-304/306 queue job (usage: sbatch --time=.. patchmem/slurm/server_job.sbatch D:ARM:SEED); GPU 0 of the allocation only
#SBATCH --job-name=pat304_q
#SBATCH --output=patchmem/logs/server/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
GPUS=0 JOBS_PER_GPU=1 bash patchmem/server_queue.sh "$1"
