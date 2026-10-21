#!/bin/bash
# PAT-307 model-agnostic error-memory plug-ins on frozen A0 (official MambaTS), ETTh2 L=720 H=96 (the paper setting; existing checkpoints).
# One job per seed: builds the forecast cache once, then every arm under both fitting protocols.
# usage: sbatch patchmem/slurm/plug_job.sbatch SEED
#SBATCH --job-name=pat307_plug
#SBATCH --output=patchmem/logs/server/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
s=$1
A="--task_name long_term_forecast --is_training 0 --root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 --dec_in 7 --c_out 7 --model_id ETTh2_720_96 --model MambaTS --data ETTh2 --features M --seed $s --seq_len 720 --label_len 48 --pred_len 96 --e_layers 5 --d_layers 2 --factor 1 --des ExpA0s$s --itr 1 --n_heads 16 --d_model 16 --dropout 0.3 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA"
S=long_term_forecast_ETTh2_720_96_MambaTS_ETTh2_ftM_sl720_ll48_pl96_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_ExpA0s${s}_0
echo "commit $(git rev-parse --short HEAD) seed $s"
for fit in train val; do for arm in base tefl pos ret retfull; do
  python -u patchmem/plug_a0.py --setting $S --arm $arm --seed $s --M 15 --fit $fit -- $A 2>&1 | grep -v Warning
done; done > patchmem/logs/server/E_plug_s$s.txt
