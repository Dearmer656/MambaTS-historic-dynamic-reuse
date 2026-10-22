#!/bin/bash
# PAT-307 fixed-length R1 (L=720, H=336 only; fair vs per-cell A0). usage: sbatch patchmem/slurm/fix_job.sbatch E|W ARM SEED   ARM: A1|T0r|T0|O|EP
#SBATCH --job-name=pat307_fix
#SBATCH --output=patchmem/logs/server/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=$1; ARM=$2; s=$3
export SEED=$s R1_LMIN=720 R1_LMAX=720 R1_HMIN=7 R1_HMAX=7 R1_VAL_GRID=720:336
case $ARM in
  A1) T=h0 ;;
  T0r) export R1_TEFL=native R1_TEFL_HC=2 R1_TEFL_RANK=64; T=TfnH2R64h0 ;;
  T0) export R1_TEFL=native R1_TEFL_HC=0 R1_TEFL_RANK=64; T=TfnH0R64h0 ;;
  O) export R1_OBS=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5; T=ObK0.5Q8h0 ;;
  EP) export R1_OBS=1 R1_OBS_VPATH=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5; T=ObK0.5Q8PVh0 ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
case $D in
  E) export DES=R1${T}F720x336s$s; S=patchmem/r1_etth2.sh ;;
  W) export DES=R1W${T}F720x336s$s; S=patchmem/r1_weather.sh ;;
  *) echo "unknown data $D"; exit 1 ;;
esac
echo "commit $(git rev-parse --short HEAD) fixed L720 H336 $D $ARM seed $s DES $DES"
bash $S > patchmem/logs/server/${D}_fix720x336_${ARM}_s$s.txt 2>&1
