#!/bin/bash
# PAT-320 v2 stage 2 for one seed (sbatch --export=S=2021 ...): arms c nopath, d path, e FreshCast on the frozen tf50 backbone.
#SBATCH --job-name=p320v2_arms
#SBATCH --output=pat_logs/p320v2/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm41,elm42,elm74
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/OpenLTM
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(echo "commit $(git rev-parse --short HEAD) seed $S host $(hostname)"; python -u pat320/v2.py --ckpt pat_logs/p320v2/joint_tf50_s$S.pth --seed $S) > pat_logs/p320v2/v2_s$S.txt 2>&1
