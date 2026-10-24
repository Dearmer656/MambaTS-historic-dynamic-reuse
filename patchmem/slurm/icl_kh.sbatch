#!/bin/bash
# PAT-307 task 6 check: offline id0 (A1, eta 1) with the error length K = the forecast's H (R1_ICL_KH=$KH), ETTh2, full 3x3 test grid.
# hyb (TEFL + one ICL step on its residuals) loads the T0-MS checkpoint (ETTh2 only) and needs KH=0 (TEFL's W1 was trained on K = Hmax
# errors): KH=0 sbatch ... SEED E hyb -> logs {D}_iclhybK7_0_s{SEED}.
# LOGTAG=x (env) appends _x to the log / DES name (e.g. LOGTAG=tvt with R1_AUDIT_FLAGS=train,val,test, exported by sbatch).
# usage: sbatch patchmem/slurm/icl_kh.sbatch SEED [E|W] [id|ridge|hyb] [row|ced]   (logs {D}_icl{V}KH0_s{SEED}, key ced: {D}_icl{V}CedKH0_s{SEED}; id runs before 2026-10-25 were named {D}_iclIdKH0_s{SEED})
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
s=${1:-3047}; D=${2:-E}; V=${3:-id}; KEY=${4:-row}; KT=$([ $KEY = ced ] && echo Ced); KH=${KH:-1}; KX=$([ $KH = 1 ] && echo KH0 || echo K7_0)${LOGTAG:+_$LOGTAG}
CK=./checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue
echo "commit $(git rev-parse --short HEAD) seed $s"
if [ $D = W ]; then CK=./checkpoints/long_term_forecast_weather_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue; A1=${CK}_R1Wh0s${s}_0/checkpoint.pth; SH=patchmem/r1_weather.sh; else A1=${CK}_R1h0s${s}_0/checkpoint.pth; SH=patchmem/r1_etth2.sh; fi
[ $V = hyb ] && A1=${CK}_R1TfmsR64h0s${s}_0/checkpoint.pth
SEED=$s R1_TEFL=ms R1_ICL=$V R1_ICL_KEY=$KEY R1_ICL_KH=$KH R1_ICL_ETA0=1 R1_ICL_ETA_FIX=1 R1_EVAL_FROM=$A1 DES=R1Icl${V}${KT}${KX}s$s EXTRA="--is_training 0" bash $SH > patchmem/logs/server/${D}_icl${V}${KT}${KX}_s$s.txt 2>&1
grep -E "R1TeflICL|eval-from|audit test\]|Traceback|Error" patchmem/logs/server/${D}_icl${V}${KT}${KX}_s$s.txt | cut -c1-400
