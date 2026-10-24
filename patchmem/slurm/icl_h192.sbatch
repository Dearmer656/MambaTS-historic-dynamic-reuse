#!/bin/bash
# PAT-307 task 6, H = 192 only (training and test): R1_HMIN = R1_HMAX = 4 patches, so the TEFL / ICL error length K = 4 (192 steps),
# L1440: 22 in-window demos, L720: 7. Val grid and test grid H = 192. ETTh2, array index = arm * 3 + seed index.
# arms: a1 (plain R1 trained at H192) | tefl (T0-MS) | tied (Wq = Wk = W1, ReLU, W2 implicit) | hyb (TEFL W2 + ICL step)
#       teflFz / tiedFz (A1 backbone of the random-H run frozen, only the low-rank map trained at H192)
#SBATCH --job-name=pat307_h192
#SBATCH --output=patchmem/logs/server/slurm_%A_%a_h192.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-17%3
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
ARMS=(a1 tefl tied hyb teflFz tiedFz); SEEDS=(3047 3048 3049)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}; s=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
export SEED=$s R1_HMIN=4 R1_HMAX=4 R1_TEST_H=192 R1_VAL_GRID=384:192,720:192,1440:192
CK=./checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue
A1=${CK}_R1h0s${s}_0/checkpoint.pth
case $ARM in
  a1)     export DES=R1H192s$s ;;
  tefl)   export R1_TEFL=ms DES=R1TfmsH192s$s ;;
  tied)   export R1_TEFL=ms R1_ICL=tied DES=R1IclTiedH192s$s ;;
  hyb)    export R1_TEFL=ms R1_ICL=hyb DES=R1IclHybH192s$s ;;
  teflFz) export R1_TEFL=ms R1_FREEZE_FROM=$A1 DES=R1TfmsFzH192s$s ;;
  tiedFz) export R1_TEFL=ms R1_ICL=tied R1_FREEZE_FROM=$A1 DES=R1IclTiedFzH192s$s ;;
esac
echo "commit $(git rev-parse --short HEAD) arm $ARM seed $s"
bash patchmem/r1_etth2.sh > patchmem/logs/server/E_H192_${ARM}_s$s.txt 2>&1
grep -E "R1 test\]|audit test\]|Traceback|Error" patchmem/logs/server/E_H192_${ARM}_s$s.txt | cut -c1-300
