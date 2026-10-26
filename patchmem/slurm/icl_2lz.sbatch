#!/bin/bash
# PAT-313 at H192: two-layer test-time update (one GD step on W1 and W2 per window; icl2l_<a1>, icl2lw1_<a1>) of the trained H192 hyb
# inference only). Runs D:SEED in parallel on ONE GPU (user 2026-10-26: elm73, two cards, tasks split). Phase lags >= K:
# W (patch 8 h) 6,9,12 = 2/3/4 days; T and E (patch 2 days) 7,14 = 2/4 weeks.
# usage: sbatch --nodelist=elm73 --gres=gpu:6000:1 patchmem/slurm/icl_sel192.sbatch "W:3047 W:3048 ..."
#SBATCH --job-name=pat313_2lz
#SBATCH --output=patchmem/logs/server/slurm_%j_2lz.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "commit $(git rev-parse --short HEAD) $(hostname) runs $1 $(date)"
O=patchmem/logs/server/icl_2lz
XA=icl2lz_0_0.3,icl2lz_0_1,icl2lz_0_3,icl2lz_0.1_0.3,icl2lz_0.1_1,icl2lz_0.1_3
one() {   # D SEED
  local D=$1 s=$2 P SH CK PH
  case $D in
    E) P=R1; SH=patchmem/r1_etth2.sh; PH=7,14; CK=./checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue ;;
    W) P=R1W; SH=patchmem/r1_weather.sh; PH=6,9,12; CK=./checkpoints/long_term_forecast_weather_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue ;;
    T) P=R1T; SH=patchmem/r1_traffic32.sh; PH=7,14; CK=./checkpoints/long_term_forecast_traffic32_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue ;;
  esac
  env SEED=$s R1_HMIN=4 R1_HMAX=4 R1_TEST_H=192 R1_TEST_L=1440 R1_VAL_GRID=1440:192 R1_TEFL=ms R1_ICL=hyb \
      R1_EVAL_FROM=${CK}_${P}IclHybH192s${s}_0/checkpoint.pth R1_AUDIT_FLAGS=test R1_AUDIT_GRID=1440:192 R1_AUDIT_PERM_REPS=0 \
      R1_ICL_XAUDIT=$XA R1_ICL_SELPHASE=$PH R1_ICL_CASEDUMP=$O/${D}_s$s DES=${P}IclHybH192s${s}2lz EXTRA="--is_training 0" bash $SH > $O/${D}_s$s.txt 2>&1
  echo "end $D s$s $(date) | $(grep -E 'audit test\] L=1440|Traceback|Error' $O/${D}_s$s.txt | cut -c1-200)"
}
for r in $1; do one ${r%%:*} ${r##*:} & sleep 20; done
wait
