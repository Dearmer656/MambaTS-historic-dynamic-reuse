#!/bin/bash
# PAT-307 A0 (official MambaTS, official per-dataset/horizon args from scripts/long_term_forecast/MambaTS_{ETTm2,WTH}.sh), L=720.
# usage: sbatch patchmem/slurm/a0_job.sbatch DATA H SEED [L=720]      DATA: ETTh2 | ETTm2 | W | T
# T (traffic32, 2026-10-25): no official 32-channel args; uses the traffic32 R1 / TEFL architecture (= Weather H96 A0 args).
# L != 720 reuses the official per-horizon args of L=720 (the official scripts only give L=720).
#SBATCH --job-name=pat307_a0
#SBATCH --output=patchmem/logs/server/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=$1; H=$2; s=$3; L=${4:-720}
case $D in
  ETTh2) if [ "$H" = 96 ]; then X="--e_layers 5 --d_model 16 --batch_size 8 --learning_rate 0.001"; elif [ "$H" = 192 ]; then X="--e_layers 3 --d_model 16 --batch_size 16 --learning_rate 0.001"; elif [ "$H" = 336 ]; then X="--e_layers 2 --d_model 32 --batch_size 32 --learning_rate 0.0005"; else X="--e_layers 1 --d_model 128 --batch_size 16 --learning_rate 0.0005"; fi
     A="--root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 --dec_in 7 --c_out 7 --model_id ETTh2_${L}_$H --data ETTh2 --n_heads 16 --dropout 0.3 $X" ;;
  ETTm2) A="--root_path ./dataset/ETT-small --data_path ETTm2.csv --enc_in 7 --dec_in 7 --c_out 7 --model_id ETTm2_${L}_$H --data ETTm2 --e_layers 3 --n_heads 16 --d_model 16 --dropout 0.3 --batch_size 16 --learning_rate 0.001" ;;
  W) if [ "$H" = 96 ]; then X="--e_layers 3 --dropout 0.2 --batch_size 8 --learning_rate 0.0005"; elif [ "$H" = 192 ]; then X="--e_layers 2 --dropout 0.3 --batch_size 16 --learning_rate 0.001"; else X="--e_layers 2 --dropout 0.3 --batch_size 8 --learning_rate 0.001"; fi
     A="--root_path ./dataset/weather --data_path weather.csv --enc_in 21 --dec_in 21 --c_out 21 --model_id weather_${L}_$H --data custom --n_heads 16 --d_model 32 --d_ff 512 $X" ;;
  T) A="--root_path ./dataset/traffic32 --data_path traffic32.csv --target 856 --enc_in 32 --dec_in 32 --c_out 32 --model_id traffic32_${L}_$H --data custom --n_heads 16 --d_model 32 --d_ff 512 --e_layers 3 --dropout 0.2 --batch_size 8 --learning_rate 0.0005" ;;
  *) echo "unknown data $D"; exit 1 ;;
esac
echo "commit $(git rev-parse --short HEAD) A0 $D L$L H$H seed $s"
python -u patchmem/run_a0.py --task_name long_term_forecast --is_training 1 --model MambaTS --features M --seq_len $L --label_len 48 --pred_len $H --d_layers 2 --factor 1 --des ExpA0s$s --itr 1 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --train_epochs 10 --patience 3 --seed $s $A > patchmem/logs/server/A0_${D}_L${L}_H${H}_s$s.txt 2>&1
