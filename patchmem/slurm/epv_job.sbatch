#!/bin/bash
# PAT-307 ErrPath variants, joint training, same variable-length R1 recipe as E_OpathV:
#   x = ErrPath-x (value Householder driven by the observed patch, the fed-back patch in the rollout; PaTH-style input-driven)
#   f = ErrPath-F (original h-driven ErrPath + FoX-style forget gate on the logits driven by the written error)
# usage: sbatch patchmem/slurm/epv_job.sbatch E|W x|f SEED
#SBATCH --job-name=pat307_epv
#SBATCH --output=patchmem/logs/server/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=$1; V=$2; s=$3
export SEED=$s R1_OBS=1 R1_OBS_VPATH=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5
case $V in
  x) export R1_OBS_VSRC=x; T=PVx0 ;;
  f) export R1_OBS_FORGET=1; T=PVF0 ;;
  *) echo "unknown variant $V"; exit 1 ;;
esac
case $D in
  E) export DES=R1ObK0.5Q8${T}s$s; S=patchmem/r1_etth2.sh ;;
  W) export DES=R1WObK0.5Q8${T}s$s; S=patchmem/r1_weather.sh ;;
  *) echo "unknown data $D"; exit 1 ;;
esac
echo "commit $(git rev-parse --short HEAD) ErrPath-$V $D seed $s DES $DES"
bash $S > patchmem/logs/server/${D}_OpathV${V}_s$s.txt 2>&1
