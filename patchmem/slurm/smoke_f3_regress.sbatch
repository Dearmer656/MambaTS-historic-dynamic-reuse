#!/bin/bash
#SBATCH --job-name=pat304_f3reg
#SBATCH --output=patchmem/logs/%j_f3reg.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
PY=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python
echo "=== test_a0corr_cross ==="; $PY -u patchmem/test_a0corr_cross.py /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch/checkpoints/long_term_forecast_ETTh2_720_96_MambaTS_ETTh2_ftM_sl720_ll48_pl96_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_ExpA0s3047_0/checkpoint.pth
echo "=== test_a0corr ==="; $PY -u patchmem/test_a0corr.py /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch/checkpoints/long_term_forecast_ETTh2_720_96_MambaTS_ETTh2_ftM_sl720_ll48_pl96_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_ExpA0s3047_0/checkpoint.pth
