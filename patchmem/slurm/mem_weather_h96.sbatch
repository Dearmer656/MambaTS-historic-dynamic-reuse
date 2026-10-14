#!/bin/bash
#SBATCH --job-name=pat304_mem
#SBATCH --output=patchmem/logs/%j_mem.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# usage: sbatch --export=ALL,SEED=3047,MODE=mem,DES=ExpMemMemS3047 patchmem/slurm/mem_weather_h96.sbatch
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
P=checkpoints/long_term_forecast_weather_720_96_MambaTS_custom_ftM_sl720_ll48_pl96_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue_
A0C_CKPT=$PWD/${P}ExpA0Ws${SEED}_0/checkpoint.pth A0C_MEM_MODE=${MODE} A0C_MEM_R=2 A0C_BACKBONE_LR=2e-5 A0C_MEM_LR=1e-3 A0C_MEM_EPOCHS=15 A0C_MEM_PATIENCE=3 \
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/run_mem.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather --data_path weather.csv --enc_in 21 --dec_in 21 --c_out 21 --model_id weather_720_96 --model MambaTS --data custom --features M --seed ${SEED} --seq_len 720 --label_len 48 --pred_len 96 --e_layers 3 --d_layers 2 --factor 1 --des ${DES} --itr 1 --n_heads 16 --d_model 32 --d_ff 512 --dropout 0.2 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --train_epochs 10 --patience 3 --batch_size 8 --learning_rate 0.0005
