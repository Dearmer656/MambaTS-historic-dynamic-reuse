#!/bin/bash
# Several a0_job.sbatch trainings in parallel on ONE GPU (user 2026-10-25: pack tasks per card; other GPU types allowed when free,
# override with sbatch --gres=gpu:TYPE:1 --cpus-per-task --mem). Same args / logs as a0_job.sbatch.
# usage: sbatch patchmem/slurm/a0_pair.sbatch DATA H "SEED:L SEED:L"
#SBATCH --job-name=pat307_a0x2
#SBATCH --output=patchmem/logs/server/slurm_%j_a0x2.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm72
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
echo "commit $(git rev-parse --short HEAD) pair $1 H$2: $3"
for p in $3; do bash patchmem/slurm/a0_job.sbatch $1 $2 ${p%%:*} ${p##*:} & sleep 30; done
wait
echo "all done $(date)"
