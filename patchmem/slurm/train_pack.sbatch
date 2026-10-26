#!/bin/bash
# PAT-307 task 6: several fixed-H R1 trainings in parallel on ONE GPU (user 2026-10-25: pack tasks per card, any free GPU type; override
# gres on the sbatch line; elm72 excluded). Each run = D:H:ARM:SEED, settings as weather_hyb.sbatch / laptop fix3 (R1_HMIN = R1_HMAX = H/48,
# val / test at H): ARM a1 | tefl | hyb | hybmean | hybperm (r1_tefl_icl.py training-time controls of hyb) | hybatt (softmax + lag
# bias initialised on the phase lags >= K, kappa 4, tau 0.25; PAT-313). A run whose log already has
# its final L=1440 line is skipped. Logs {D}_H{H}_{ARM}_s{SEED}.txt.  usage: sbatch --gres=gpu:TYPE:1 patchmem/slurm/train_pack.sbatch "W:96:hybmean:3047 ..."
#SBATCH --job-name=pat307_tpack
#SBATCH --output=patchmem/logs/server/slurm_%j_tpack.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm72
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "commit $(git rev-parse --short HEAD) runs $1 on $(hostname) $(date)"
one() {   # one D H ARM SEED
  local D=$1 H=$2 A=$3 s=$4 Hp=$(( $2 / 48 )) P SH
  case $D in E) P=R1; SH=patchmem/r1_etth2.sh ;; W) P=R1W; SH=patchmem/r1_weather.sh ;; T) P=R1T; SH=patchmem/r1_traffic32.sh ;; esac
  local LOG=patchmem/logs/server/${D}_H${H}_${A}_s$s.txt
  local DONE="R1 test\] L=1440 H= *$H"; [ $A = a1 ] || DONE="audit test\] L=1440 H= *$H"
  if [ -f $LOG ] && grep -q "$DONE" $LOG; then echo "skip $LOG"; return; fi
  local X=(SEED=$s R1_HMIN=$Hp R1_HMAX=$Hp R1_TEST_H=$H R1_VAL_GRID=384:$H,720:$H,1440:$H)
  case $A in
    a1)      X+=(DES=${P}H${H}s$s) ;;
    tefl)    X+=(R1_TEFL=ms DES=${P}TfmsH${H}s$s) ;;
    hyb)     X+=(R1_TEFL=ms R1_ICL=hyb DES=${P}IclHybH${H}s$s) ;;
    hybmean) X+=(R1_TEFL=ms R1_ICL=hybmean DES=${P}IclHybMeanH${H}s$s) ;;
    hybperm) X+=(R1_TEFL=ms R1_ICL=hybperm DES=${P}IclHybPermH${H}s$s) ;;
    hybatt)  local step=$([ $D = W ] && echo 3 || echo 7)                               # phase: W day = 3 patches; T, E week ~ 7 patches (2 weeks)
             local pri=$(seq $step $step 29 | awk -v k=$Hp '$1 >= k' | paste -sd,)
             X+=(R1_TEFL=ms R1_ICL=hybatt R1_ICL_LAGPRIOR=$pri R1_ICL_KAPPA=4 R1_ICL_TAU=0.25 DES=${P}IclHybAttH${H}s$s) ;;
  esac
  echo "start $LOG $(date)"
  env "${X[@]}" bash $SH > $LOG 2>&1
  echo "end $LOG $(date) | $(grep -E "R1 test\] L=1440|Traceback" $LOG | cut -c1-120)"
}
for r in $1; do IFS=: read D H A s <<< "$r"; one $D $H $A $s & sleep 30; done
wait
echo "all done $(date)"
