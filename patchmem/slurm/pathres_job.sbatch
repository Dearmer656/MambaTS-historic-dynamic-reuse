#!/bin/bash
# PAT-307 R1PathRes (path-transported lookback errors, no attention; patchmem/r1_pathres.py) on ETTh2 P=48, same recipe as O / T0.
# usage: sbatch patchmem/slurm/pathres_job.sbatch ARM SEED      ARM: Pr (Householder path) | PrI (H = I ablation)
#SBATCH --job-name=pat307_pathres
#SBATCH --output=patchmem/logs/server/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
ARM=$1; s=$2
export SEED=$s R1_PATHRES=1 R1_PATHRES_RANK=32
case $ARM in
  Pr)  export DES=R1PrR32h0s$s ;;
  PrI) export R1_PATHRES_NOH=1 DES=R1PrIR32h0s$s ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
echo "commit $(git rev-parse --short HEAD) arm $ARM seed $s"
bash patchmem/r1_etth2.sh > patchmem/logs/server/E_${ARM}_s$s.txt 2>&1
