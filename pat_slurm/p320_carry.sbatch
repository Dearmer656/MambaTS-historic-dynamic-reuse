#!/bin/bash
# PAT-320 stage 2 on the frozen joint backbones: per seed s (backbone seed = head seed), arm b (ridge), d (path), c (nopath).
# Three seed streams run in parallel on one GPU.
#SBATCH --job-name=p320_carry
#SBATCH --output=pat_logs/p320/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm41,elm42,elm74
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/OpenLTM
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
for s in 2021 2022 2023; do
  (
  CK=pat_logs/p320/joint_s$s.pth; H="commit $(git rev-parse --short HEAD) seed $s host $(hostname)"
  (echo $H; python -u pat320/ridge.py --ckpt $CK) > pat_logs/p320/ridge_s$s.txt 2>&1
  for A in path nopath; do (echo $H; python -u pat320/carry.py --ckpt $CK --arm $A --seed $s) > pat_logs/p320/carry_${A}_s$s.txt 2>&1; done
  ) &
done
wait
