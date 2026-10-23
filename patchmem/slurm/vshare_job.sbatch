#!/bin/bash
# PAT-307 ErrPath-shared: O-PathV with the Householder in the P-dim patch space, the same reflection for every variable (no cross-variable
# mixing; W_wv -> P instead of -> C*P). Same variable-length R1 recipe as E_OpathV / W_OpathV. usage: sbatch patchmem/slurm/vshare_job.sbatch E|W|T SEED
#SBATCH --job-name=pat307_vshare
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
export SEED=$s R1_OBS=1 R1_OBS_VPATH=1 R1_OBS_VSHARE=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5
case $D in
  E) export DES=R1ObK0.5Q8PVsh0s$s; S=patchmem/r1_etth2.sh ;;
  W) export DES=R1WObK0.5Q8PVsh0s$s; S=patchmem/r1_weather.sh ;;
  T) export DES=R1TObK0.5Q8PVsh0v2s$s; S=patchmem/r1_traffic32.sh ;;
  *) echo "unknown data $D"; exit 1 ;;
esac
echo "commit $(git rev-parse --short HEAD) ErrPath-shared $D seed $s DES $DES"
bash $S > patchmem/logs/server/${D}_OpathVs_s$s.txt 2>&1
