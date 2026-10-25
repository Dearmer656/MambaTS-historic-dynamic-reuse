#!/bin/bash
# PAT-313 point 6: select-then-adapt audits (iclsel, iclsel1, iclselfix_*) of the trained hyb (H96, L1440, test, inference only),
# vs its own ICL term (icloff gives the gain). W and T x 3 seeds in parallel on ONE GPU.
# usage: sbatch --gres=gpu:TYPE:1 patchmem/slurm/icl_sel.sbatch
#SBATCH --job-name=pat313_sel
#SBATCH --output=patchmem/logs/server/slurm_%j_sel.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm72
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "commit $(git rev-parse --short HEAD) $(hostname) $(date)"
O=patchmem/logs/server/icl_sel
SEL=iclsel,iclsel1,iclselfix_all,iclselfix_recent4,iclselfix_phase,iclselfix_topk8
one() {   # D SEED
  local D=$1 s=$2 P SH CK XA PER PH
  case $D in
    W) P=R1W; SH=patchmem/r1_weather.sh; PER=144; PH=3,6,9; XA=$SEL
       CK=./checkpoints/long_term_forecast_weather_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue ;;
    T) P=R1T; SH=patchmem/r1_traffic32.sh; PER=24; PH=7,14; XA=$SEL
       CK=./checkpoints/long_term_forecast_traffic32_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue ;;
  esac
  env SEED=$s R1_HMIN=2 R1_HMAX=2 R1_TEST_H=96 R1_TEST_L=1440 R1_VAL_GRID=1440:96 R1_TEFL=ms R1_ICL=hyb \
      R1_EVAL_FROM=${CK}_${P}IclHybH96s${s}_0/checkpoint.pth R1_AUDIT_FLAGS=test R1_AUDIT_GRID=1440:96 R1_AUDIT_PERM_REPS=0 \
      R1_ICL_XAUDIT=$XA R1_ICL_PERIOD=$PER R1_ICL_SELPHASE=$PH R1_ICL_CASEDUMP=$O/${D}_s$s DES=${P}IclHybH96s${s}sel EXTRA="--is_training 0" bash $SH > $O/${D}_s$s.txt 2>&1
  echo "end $D s$s $(date) | $(grep -E 'audit test\] L=1440|Traceback|Error' $O/${D}_s$s.txt | cut -c1-200)"
}
for D in W T; do for s in 3047 3048 3049; do one $D $s & sleep 20; done; done
wait
ls -la $O/*.pt
