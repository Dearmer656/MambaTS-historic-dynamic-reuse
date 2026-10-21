#!/bin/bash
# PAT-307 plug-ins on frozen A0 for ETTm2 / Weather (checkpoints from patchmem/slurm/a0_job.sbatch, same official args), L=720.
# usage: sbatch patchmem/slurm/plug_job2.sbatch DATA H SEED [ARMS]      DATA: ETTm2 | W
#SBATCH --job-name=pat307_plug2
#SBATCH --output=patchmem/logs/server/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=$1; H=$2; s=$3; ARMS=${4:-"base tefl pos posfull ret retfull"}
case $D in
  ETTm2) A="--root_path ./dataset/ETT-small --data_path ETTm2.csv --enc_in 7 --dec_in 7 --c_out 7 --model_id ETTm2_720_$H --data ETTm2 --e_layers 3 --n_heads 16 --d_model 16 --dropout 0.3 --batch_size 16 --learning_rate 0.001" ;;
  W) if [ "$H" = 96 ]; then X="--e_layers 3 --dropout 0.2"; else X="--e_layers 2 --dropout 0.3"; fi
     A="--root_path ./dataset/weather --data_path weather.csv --enc_in 21 --dec_in 21 --c_out 21 --model_id weather_720_$H --data custom --n_heads 16 --d_model 32 --d_ff 512 $X" ;;
esac
A="--task_name long_term_forecast --is_training 0 --model MambaTS --features M --seq_len 720 --label_len 48 --pred_len $H --d_layers 2 --factor 1 --des ExpA0s$s --itr 1 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --seed $s $A"
S=$(ls checkpoints | grep "_$( [ $D = W ] && echo weather || echo ETTm2 )_720_${H}_MambaTS_" | grep "_ExpA0s${s}_0$")
[ -n "$S" ] || { echo "no checkpoint for $D H$H s$s"; exit 1; }
echo "commit $(git rev-parse --short HEAD) $D H$H seed $s ckpt $S"
for fit in train val; do for arm in $ARMS; do
  python -u patchmem/plug_a0.py --setting $S --arm $arm --seed $s --M 15 --fit $fit -- $A 2>&1 | grep -v Warning
done; done > patchmem/logs/server/${D}_H${H}_plug_s$s.txt
