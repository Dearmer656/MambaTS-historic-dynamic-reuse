#!/bin/bash
# PAT-307 O-plain (same recipe as O = R1ObK0.5Q8): one head, correction = gamma * sum_j a_j e_j (RATL-style, no W / gate).
# usage: sbatch patchmem/slurm/oplain_job.sbatch SEED [E|W]      E = ETTh2 (default), W = Weather
#SBATCH --job-name=pat307_oplain
#SBATCH --output=patchmem/logs/server/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
s=$1; D=${2:-E}
export SEED=$s R1_OBS=1 R1_OBS_PLAIN=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5
echo "commit $(git rev-parse --short HEAD) O-plain data $D seed $s"
case $D in
  E) DES=R1ObK0.5Q8Plh0s$s bash patchmem/r1_etth2.sh > patchmem/logs/server/E_Oplain_s$s.txt 2>&1 ;;
  W) DES=R1WObK0.5Q8Plh0s$s bash patchmem/r1_weather.sh > patchmem/logs/server/W_Oplain_s$s.txt 2>&1 ;;
  *) echo "unknown data $D"; exit 1 ;;
esac
