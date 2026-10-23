#!/bin/bash
# PAT-307 frozen R1: the backbone of the trained plain R1 (A1, same seed) is loaded and frozen (eval mode, fixed eval scan order, no VPT
# cost update); only the correction module is trained, same variable-length recipe / early stopping / test grid as the joint arms.
# usage: sbatch patchmem/slurm/frz_job.sbatch E|W ARM SEED   ARM: O|EP|T0|T0r
#SBATCH --job-name=pat307_frz
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
export SEED=$s
case $ARM in
  O) export R1_OBS=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5 ;;
  EP) export R1_OBS=1 R1_OBS_VPATH=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5 ;;
  T0) export R1_TEFL=native R1_TEFL_HC=0 R1_TEFL_RANK=64 ;;
  T0r) export R1_TEFL=native R1_TEFL_HC=2 R1_TEFL_RANK=64 ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
case $D in
  E) export DES=R1Fz${ARM}s$s R1_FREEZE_FROM=checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_R1h0s${s}_0/checkpoint.pth; S=patchmem/r1_etth2.sh ;;
  W) export DES=R1WFz${ARM}s$s R1_FREEZE_FROM=checkpoints/long_term_forecast_weather_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue_R1Wh0s${s}_0/checkpoint.pth; S=patchmem/r1_weather.sh ;;
  *) echo "unknown data $D"; exit 1 ;;
esac
test -f $R1_FREEZE_FROM || { echo "missing $R1_FREEZE_FROM"; exit 1; }
echo "commit $(git rev-parse --short HEAD) frozen-R1 $D $ARM seed $s DES $DES"
bash $S > patchmem/logs/server/${D}_Fz_${ARM}_s$s.txt 2>&1
