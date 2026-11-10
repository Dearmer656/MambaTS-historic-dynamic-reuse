#!/bin/bash
# PAT-320 stage 1: joint channel-independent Timer-XL on 7 benchmarks, 3 seeds packed on one GPU.
#SBATCH --job-name=p320_joint
#SBATCH --output=pat_logs/p320/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm41,elm42,elm74
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/OpenLTM
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
for s in 2021 2022 2023; do
  (echo "commit $(git rev-parse --short HEAD) seed $s host $(hostname)"; python -u pat320/joint.py --seed $s) > pat_logs/p320/joint_s$s.txt 2>&1 &
done
wait
