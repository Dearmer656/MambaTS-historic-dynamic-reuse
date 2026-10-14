#!/bin/bash
#SBATCH --job-name=pat304_rlw
#SBATCH --output=patchmem/logs/%j_rlw.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=18:00:00
# usage: sbatch --export=ALL,SEED=3047 patchmem/slurm/diag_retrieval_weather.sbatch   (trainable Q/K retrieval on frozen Weather A0)
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
P=checkpoints/long_term_forecast_weather_720_96_MambaTS_custom_ftM_sl720_ll48_pl96_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue_
A0C_MODE=cross A0C_CKPT=$PWD/${P}ExpA0Ws${SEED}_0/checkpoint.pth A0C_CORR=none A0C_INNOV=6 A0C_RBS=8 A0C_TRSTRIDE=12 A0C_EVSTRIDE=2 /cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/run_a0corr.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather --data_path weather.csv --enc_in 21 --dec_in 21 --c_out 21 --model_id weather_720_96 --model MambaTS --data custom --features M --seed ${SEED} --seq_len 720 --label_len 48 --pred_len 96 --e_layers 3 --d_layers 2 --factor 1 --des ExpDiagW --itr 1 --n_heads 16 --d_model 32 --d_ff 512 --dropout 0.2 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --train_epochs 10 --patience 3 --batch_size 32 --learning_rate 0.0005
