#!/bin/bash
#SBATCH --job-name=pat304_a0cw
#SBATCH --output=patchmem/logs/%j_a0cw.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# usage: sbatch --export=ALL,SEED=3047,DES=ExpA0CWs3047,A0C_CORR=gated[,A0C_CKPT=<A0 checkpoint.pth>] patchmem/slurm/a0corr_weather_h96.sbatch
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/run_a0corr.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/weather --data_path weather.csv --enc_in 21 --dec_in 21 --c_out 21 --model_id weather_720_96 --model MambaTS --data custom --features M --seed ${SEED:-3047} --seq_len 720 --label_len 48 --pred_len 96 --e_layers 3 --d_layers 2 --factor 1 --des ${DES:-ExpA0CW} --itr 1 --n_heads 16 --d_model 32 --d_ff 512 --dropout 0.2 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --train_epochs 10 --patience 3 --batch_size 8 --learning_rate 0.0005
