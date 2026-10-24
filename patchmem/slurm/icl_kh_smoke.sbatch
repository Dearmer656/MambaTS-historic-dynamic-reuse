#!/bin/bash
# PAT-307 task 6 smoke for the K = H trained arms (R1_ICL_KH=1): TEFL-KH (hyb, eta fixed 0), tied-KH, hyb-KH, tied-KH on a frozen A1.
#SBATCH --job-name=pat307_khsmk
#SBATCH --output=patchmem/logs/server/slurm_%j_khsmk.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export SEED=3047 EPOCHS=1 R1_MAX_ITERS=30 R1_VAL_STRIDE=32 R1_AUDIT_GRID=1440:96,720:96 R1_TEST_L=720,1440 R1_TEST_H=96,336 R1_TEFL=ms R1_ICL_KH=1
CK=./checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue
echo "commit $(git rev-parse --short HEAD) (+ working tree)"
f() { grep -E "R1TeflICL|frozen\] backbone|Epoch|audit test\]|Error|Traceback|nan" | cut -c1-300; }
R1_ICL=hyb R1_ICL_ETA0=0 R1_ICL_ETA_FIX=1 DES=smkKHtefl bash patchmem/r1_etth2.sh 2>&1 | f
R1_ICL=tied DES=smkKHtied bash patchmem/r1_etth2.sh 2>&1 | f
R1_ICL=hyb DES=smkKHhyb bash patchmem/r1_etth2.sh 2>&1 | f
R1_ICL=tied R1_FREEZE_FROM=${CK}_R1h0s3047_0/checkpoint.pth DES=smkKHtiedFz bash patchmem/r1_etth2.sh 2>&1 | f
