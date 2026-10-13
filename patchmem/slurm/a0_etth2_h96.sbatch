#!/bin/bash
#SBATCH --job-name=pat304_a0
#SBATCH --output=patchmem/logs/%j_a0.txt
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3:30:00
# usage: sbatch --export=ALL,SEED=3047,DES=ExpA0 patchmem/slurm/a0_etth2_h96.sbatch   (A0 = official MambaTS, ETTh2, H=96, official script args)
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
/cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/run_a0.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 --dec_in 7 --c_out 7 --model_id ETTh2_720_96 --model MambaTS --data ETTh2 --features M --seed ${SEED:-3047} --seq_len 720 --label_len 48 --pred_len 96 --e_layers 5 --d_layers 2 --factor 1 --des ${DES:-ExpA0} --itr 1 --n_heads 16 --d_model 16 --dropout 0.3 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --train_epochs 10 --patience 3 --batch_size 8 --learning_rate 0.001
