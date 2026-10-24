#!/bin/bash
# PAT-307 task 6 smoke: R1TeflICL tied / hyb / qk training, tied on a frozen A1 (30 iters) + offline id (A1) / hyb0 (T0-MS) test-only from checkpoints, ETTh2 s3047.
#SBATCH --job-name=pat307_iclsmk
#SBATCH --output=patchmem/logs/server/slurm_%j_iclsmk.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export SEED=3047 EPOCHS=1 R1_MAX_ITERS=30 R1_VAL_STRIDE=32 R1_AUDIT_GRID=1440:336,720:336 R1_TEST_L=720,1440 R1_TEST_H=96,336 R1_TEFL=ms
CK=./checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue
echo "commit $(git rev-parse --short HEAD)"
: # joint arms checked in job 6006
R1_ICL=tied R1_FREEZE_FROM=${CK}_R1h0s3047_0/checkpoint.pth DES=smkICLtiedFz bash patchmem/r1_etth2.sh 2>&1 | grep -E "R1TeflICL|frozen|Epoch|audit test\]|Error|Traceback|nan"
R1_ICL=id R1_ICL_ETA0=1 R1_ICL_ETA_FIX=1 R1_EVAL_FROM=${CK}_R1h0s3047_0/checkpoint.pth DES=smkICLid0 EXTRA="--is_training 0" bash patchmem/r1_etth2.sh 2>&1 | grep -E "R1TeflICL|eval-from|R1 test\]|audit test\]|Error|Traceback"
R1_ICL=hyb R1_ICL_ETA0=1 R1_ICL_ETA_FIX=1 R1_EVAL_FROM=${CK}_R1TfmsR64h0s3047_0/checkpoint.pth DES=smkICLhyb0 EXTRA="--is_training 0" bash patchmem/r1_etth2.sh 2>&1 | grep -E "R1TeflICL|eval-from|R1 test\]|audit test\]|Error|Traceback"
