#!/bin/bash
# PAT-313 decoupled lookback (v2): forecast and TEFL query exactly as at L = 384 (the last 384 steps); demos = TEFL's own (input, target)
# pairs from every earlier 384-step window inside the L_ctx input (R1_ICL_LB=384, lb_pairs), demo stride DS. usage: ... "E:3047 ..." <Lctx> "<DS ...>"
#SBATCH --job-name=pat313_lb
#SBATCH --output=patchmem/logs/server/slurm_%j_lb.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm72
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "commit $(git rev-parse --short HEAD) $(hostname) runs $1 $(date)"
XA=icl2l_0.1,icl2l_0.3,icl2l_1,icl2l_3
LC=$2
one() {   # D SEED
  local D=$1 s=$2 P SH CK PH
  case $D in
    E) P=R1; SH=patchmem/r1_etth2.sh; PH=7,14; CK=./checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue ;;
    W) P=R1W; SH=patchmem/r1_weather.sh; PH=6,9,12; CK=./checkpoints/long_term_forecast_weather_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue ;;
    T) P=R1T; SH=patchmem/r1_traffic32.sh; PH=7,14; CK=./checkpoints/long_term_forecast_traffic32_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue ;;
  esac
  env SEED=$s R1_HMIN=4 R1_HMAX=4 R1_TEST_H=192 R1_TEST_L=$LC R1_VAL_GRID=1440:192 R1_TEFL=ms R1_ICL=hyb \
      R1_EVAL_FROM=${CK}_${P}IclHybH192s${s}_0/checkpoint.pth R1_AUDIT_FLAGS=test R1_AUDIT_GRID=$LC:192 R1_AUDIT_PERM_REPS=0 \
      R1_ICL_XAUDIT=$XA R1_ICL_LB=384 R1_ICL_DSTRIDE=$DS R1_ICL_CASEDUMP=$O/${D}_s$s DES=${P}IclHybH192s${s}lbv2${LC}s$DS EXTRA="--is_training 0" bash $SH > $O/${D}_s$s.txt 2>&1
  echo "end $D s$s $(date) | $(grep -E 'audit test\] L=1440|Traceback|Error' $O/${D}_s$s.txt | cut -c1-200)"
}
for DS in ${3:-1}; do O=patchmem/logs/server/icl_lbv2/L${LC}s$DS; mkdir -p $O; for r in $1; do one ${r%%:*} ${r##*:} & sleep 20; done; done
wait
