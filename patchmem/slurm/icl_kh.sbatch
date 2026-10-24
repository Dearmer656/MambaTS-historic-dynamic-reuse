#!/bin/bash
# PAT-307 task 6 check: offline id0 (A1, eta 1) with the error length K = the forecast's H (R1_ICL_KH=1), ETTh2, full 3x3 test grid.
# usage: sbatch patchmem/slurm/icl_kh.sbatch SEED [E|W]
#SBATCH --job-name=pat307_iclkh
#SBATCH --output=patchmem/logs/server/slurm_%j_iclkh.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
s=${1:-3047}; D=${2:-E}
CK=./checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue
echo "commit $(git rev-parse --short HEAD) seed $s"
if [ $D = W ]; then CK=./checkpoints/long_term_forecast_weather_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue; A1=${CK}_R1Wh0s${s}_0/checkpoint.pth; SH=patchmem/r1_weather.sh; else A1=${CK}_R1h0s${s}_0/checkpoint.pth; SH=patchmem/r1_etth2.sh; fi
SEED=$s R1_TEFL=ms R1_ICL=id R1_ICL_KH=1 R1_ICL_ETA0=1 R1_ICL_ETA_FIX=1 R1_EVAL_FROM=$A1 DES=R1IclIdKH0s$s EXTRA="--is_training 0" bash $SH > patchmem/logs/server/${D}_iclIdKH0_s$s.txt 2>&1
grep -E "R1TeflICL|eval-from|audit test\]|Traceback|Error" patchmem/logs/server/${D}_iclIdKH0_s$s.txt | cut -c1-400
