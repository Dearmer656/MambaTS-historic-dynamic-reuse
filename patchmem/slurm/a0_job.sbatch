#!/bin/bash
# PAT-307 A0 (official MambaTS, official per-dataset/horizon args from scripts/long_term_forecast/MambaTS_{ETTm2,WTH}.sh), L=720.
# usage: sbatch patchmem/slurm/a0_job.sbatch DATA H SEED      DATA: ETTm2 | W
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
D=$1; H=$2; s=$3
case $D in
  ETTm2) A="--root_path ./dataset/ETT-small --data_path ETTm2.csv --enc_in 7 --dec_in 7 --c_out 7 --model_id ETTm2_720_$H --data ETTm2 --e_layers 3 --n_heads 16 --d_model 16 --dropout 0.3 --batch_size 16 --learning_rate 0.001" ;;
  W) if [ "$H" = 96 ]; then X="--e_layers 3 --dropout 0.2 --batch_size 8 --learning_rate 0.0005"; elif [ "$H" = 192 ]; then X="--e_layers 2 --dropout 0.3 --batch_size 16 --learning_rate 0.001"; else X="--e_layers 2 --dropout 0.3 --batch_size 8 --learning_rate 0.001"; fi
     A="--root_path ./dataset/weather --data_path weather.csv --enc_in 21 --dec_in 21 --c_out 21 --model_id weather_720_$H --data custom --n_heads 16 --d_model 32 --d_ff 512 $X" ;;
  *) echo "unknown data $D"; exit 1 ;;
esac
echo "commit $(git rev-parse --short HEAD) A0 $D H$H seed $s"
python -u patchmem/run_a0.py --task_name long_term_forecast --is_training 1 --model MambaTS --features M --seq_len 720 --label_len 48 --pred_len $H --d_layers 2 --factor 1 --des ExpA0s$s --itr 1 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --train_epochs 10 --patience 3 --seed $s $A > patchmem/logs/server/A0_${D}_H${H}_s$s.txt 2>&1
