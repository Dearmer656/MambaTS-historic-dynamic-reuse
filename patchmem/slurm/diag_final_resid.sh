#!/bin/bash
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
P=checkpoints/long_term_forecast_ETTh2_720_96_MambaTS_ETTh2_ftM_sl720_ll48_pl96_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue_
for s in 3047 3048 3049; do for arm in A0; do
  sh=0; [ $arm = F2shuf ] && sh=1
  echo "=== $arm s$s"
  A0C_CKPT=$PWD/${P}ExpA0s${s}_0/checkpoint.pth A0C_CORR=none A0C_INNOV=3 A0C_SHUFFLE=$sh /cl/work5/hongyu-s/conda/envs/mamba_iso/bin/python -u patchmem/run_a0corr.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small --data_path ETTh2.csv --enc_in 7 --dec_in 7 --c_out 7 --model_id ETTh2_720_96 --model MambaTS --data ETTh2 --features M --seed $s --seq_len 720 --label_len 48 --pred_len 96 --e_layers 5 --d_layers 2 --factor 1 --des ExpDiag --itr 1 --n_heads 16 --d_model 16 --dropout 0.3 --patch_len 48 --stride 48 --VPT_mode 1 --ATSP_solver SA --train_epochs 10 --patience 3 --batch_size 8 --learning_rate 0.001 2>&1 | command grep -a -E "^\[final|aux next|Traceback|Error"
done; done
