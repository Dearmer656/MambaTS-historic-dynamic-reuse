#!/bin/bash
# PAT-313 TEFL original setting on A0 (patchmem/exp_a0tefl.py): joint training from scratch, warm-up 3 epochs (MSE + SF) + joint 12 (AdamW),
# context window = L, error window = H. Official per-dataset/horizon A0 args (as a0_job.sbatch). Runs "D:H:SEED[:arm]" in parallel on one GPU;
# arm tefl (default) | a0 (plain A0 baseline via run_a0.py, for missing baselines). usage: bash/sbatch a0tefl_pack.sbatch "E:192:3047 ..." [L=384]
#SBATCH --job-name=pat313_a0tefl
#SBATCH --output=patchmem/logs/server/slurm_%j_a0tefl.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=elm72
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=${2:-384}
echo "commit $(git rev-parse --short HEAD) $(hostname) runs $1 L$L $(date)"
one() {   # D H SEED ARM
  local D=$1 H=$2 s=$3 arm=${4:-tefl} X A
  case $D in
    E) if [ "$H" = 96 ]; then X="--e_layers 5 --d_model 16 --batch_size 8 --learning_rate 0.001"; elif [ "$H" = 192 ]; then X="--e_layers 3 --d_model 16 --batch_size 16 --learning_rate 0.001"; else X="--e_layers 2 --d_model 32 --batch_size 32 --learning_rate 0.0005"; fi
       A="--root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 --dec_in 7 --c_out 7 --model_id ETTh2_${L}_$H --data ETTh2 --n_heads 16 --dropout 0.3 $X" ;;
    W) if [ "$H" = 96 ]; then X="--e_layers 3 --dropout 0.2 --batch_size 8 --learning_rate 0.0005"; elif [ "$H" = 192 ]; then X="--e_layers 2 --dropout 0.3 --batch_size 16 --learning_rate 0.001"; else X="--e_layers 2 --dropout 0.3 --batch_size 8 --learning_rate 0.001"; fi
       A="--root_path ./dataset/weather --data_path weather.csv --enc_in 21 --dec_in 21 --c_out 21 --model_id weather_${L}_$H --data custom --n_heads 16 --d_model 32 --d_ff 512 $X" ;;
    T) A="--root_path ./dataset/traffic32 --data_path traffic32.csv --target 856 --enc_in 32 --dec_in 32 --c_out 32 --model_id traffic32_${L}_$H --data custom --n_heads 16 --d_model 32 --d_ff 512 --e_layers 3 --dropout 0.2 --batch_size 8 --learning_rate 0.0005" ;;
  esac
  local C="--task_name long_term_forecast --is_training 1 --model MambaTS --features M --seq_len $L --label_len 48 --pred_len $H --d_layers 2 --factor 1 --itr 1 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --train_epochs 10 --patience 3 --seed $s $A"
  if [ "$arm" = test ]; then
    python -u patchmem/run_a0tefl.py ${C/--is_training 1/--is_training 0} --des A0Tefls$s > patchmem/logs/server/A0TEFL_${D}_L${L}_H${H}_s${s}_test.txt 2>&1
    echo "end test $D H$H s$s $(date) | $(grep -hE "^mse:|g removed|Traceback" patchmem/logs/server/A0TEFL_${D}_L${L}_H${H}_s${s}_test.txt | tail -2 | tr "\n" " ")"
  elif [ "$arm" = a0 ]; then
    python -u patchmem/run_a0.py $C --des ExpA0s$s > patchmem/logs/server/A0_${D}_L${L}_H${H}_s$s.txt 2>&1
    echo "end a0 $D H$H s$s $(date) | $(grep -h '^mse:' patchmem/logs/server/A0_${D}_L${L}_H${H}_s$s.txt | tail -1)"
  else
    python -u patchmem/run_a0tefl.py $C --des A0Tefls$s > patchmem/logs/server/A0TEFL_${D}_L${L}_H${H}_s$s.txt 2>&1
    echo "end tefl $D H$H s$s $(date) | $(grep -hE '^mse:|g removed|Traceback' patchmem/logs/server/A0TEFL_${D}_L${L}_H${H}_s$s.txt | tail -2 | tr '\n' ' ')"
  fi
}
for r in $1; do IFS=: read D H s arm <<< "$r"; one $D $H $s $arm & sleep 20; done
wait
