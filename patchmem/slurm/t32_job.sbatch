#!/bin/bash
# PAT-307 long-range test of the error path on Traffic-32 (hourly, weekly period): variable-length R1 (L 384..1440, H 96..336), same
# arms and settings as ETTh2 / Weather. usage: sbatch patchmem/slurm/t32_job.sbatch ARM SEED   ARM: A1|T0|T0r|O|EP
#SBATCH --job-name=pat307_t32
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
export SEED=$s
case $ARM in
  A1) T=h0 ;;
  T0) export R1_TEFL=native R1_TEFL_HC=0 R1_TEFL_RANK=64; T=TfnH0R64h0 ;;
  T0r) export R1_TEFL=native R1_TEFL_HC=2 R1_TEFL_RANK=64; T=TfnH2R64h0 ;;
  O) export R1_OBS=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5; T=ObK0.5Q8h0 ;;
  EP) export R1_OBS=1 R1_OBS_VPATH=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5; T=ObK0.5Q8PVh0 ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
export DES=R1T${T}v2s$s
echo "commit $(git rev-parse --short HEAD) traffic32 $ARM seed $s DES $DES"
bash patchmem/r1_traffic32.sh > patchmem/logs/server/T2_${ARM}_s$s.txt 2>&1
