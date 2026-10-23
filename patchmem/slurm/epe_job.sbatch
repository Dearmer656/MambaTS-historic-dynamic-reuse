#!/bin/bash
# PAT-307 ErrPath-e: O-PathV with the value Householder driven by the newly written error (w = normalize(W_wv e_i), C*P space), H = I in
# the rollout. Same variable-length R1 recipe as E_OpathV. usage: sbatch patchmem/slurm/epe_job.sbatch E|W SEED
#SBATCH --job-name=pat307_epe
#SBATCH --output=patchmem/logs/server/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=$1; s=$2
export SEED=$s R1_OBS=1 R1_OBS_VPATH=1 R1_OBS_VSRC=e R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5
case $D in
  E) export DES=R1ObK0.5Q8PVe0s$s; S=patchmem/r1_etth2.sh ;;
  W) export DES=R1WObK0.5Q8PVe0s$s; S=patchmem/r1_weather.sh ;;
  *) echo "unknown data $D"; exit 1 ;;
esac
echo "commit $(git rev-parse --short HEAD) ErrPath-e $D seed $s DES $DES"
bash $S > patchmem/logs/server/${D}_OpathVe_s$s.txt 2>&1
