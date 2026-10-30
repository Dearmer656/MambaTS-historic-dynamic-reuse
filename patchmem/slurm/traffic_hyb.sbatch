#!/bin/bash
# PAT-307 task 6, Traffic-32 check (third dataset): fixed-H training / val / test, arms a1 (plain R1) | tefl (T0-MS)
# | hyb (TEFL W2 + ICL step, eta learned). Same settings as weather_hyb.sbatch (job 7481), H = 96 (K = 2) only for now.
# 3090 array, one task per seed, arms run sequentially.
# A run whose log already has its final L=1440 line is skipped on a resubmit. usage: sbatch --array=0-2 patchmem/slurm/traffic_hyb.sbatch
#SBATCH --job-name=pat307_thyb
#SBATCH --output=patchmem/logs/server/slurm_%A_%a_thyb.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
echo "commit $(git rev-parse --short HEAD) $(date)"
one() {   # one SEED H ARM
  local s=$1 H=$2 A=$3 Hp=$(( $2 / 48 ))
  local LOG=patchmem/logs/server/T_H${H}_${A}_s$s.txt
  local DONE="R1 test\] L=1440 H= *$H"; [ $A = a1 ] || DONE="audit test\] L=1440 H= *$H"
  if [ -f $LOG ] && grep -q "$DONE" $LOG; then echo "skip $LOG"; return; fi
  local X=(SEED=$s R1_HMIN=$Hp R1_HMAX=$Hp R1_TEST_H=$H R1_VAL_GRID=384:$H,720:$H,1440:$H)
  case $A in
    a1)   X+=(DES=R1TH${H}s$s) ;;
    tefl) X+=(R1_TEFL=ms DES=R1TTfmsH${H}s$s) ;;
    hyb)  X+=(R1_TEFL=ms R1_ICL=hyb DES=R1TIclHybH${H}s$s) ;;
  esac
  echo "start $LOG $(date)"
  env "${X[@]}" bash patchmem/r1_traffic32.sh > $LOG 2>&1
  echo "end $LOG $(date) | $(grep -E "R1 test\] L=1440|Traceback" $LOG | cut -c1-120)"
}
seed() { for H in 96; do for A in a1 tefl hyb; do one $1 $H $A; done; done; }
S=(3047 3048 3049); seed ${S[$SLURM_ARRAY_TASK_ID]}; echo "all done $(date)"
