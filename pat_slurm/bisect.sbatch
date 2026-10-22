#!/bin/bash
# PAT-311 recipe bisect for ctrl vs the official Timer-XL. usage: sbatch pat_slurm/bisect.sbatch SEED VARIANT   VARIANT: A|B|C
#   A: L_tf only, official windows, select on valid_last (= the official recipe re-implemented)   B: A + L_roll (768-future windows)
#   C: L_tf + L_roll, select on valid_last (B differs from the submitted ctrl only by the selection metric)
#SBATCH --job-name=pat311_bis
#SBATCH --output=pat_logs/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/OpenLTM
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
s=$1; V=$2
case $V in
  A) X="--no_roll --sel last --tag _bisA" ;;
  C) X="--sel last --tag _bisC" ;;
esac
python -u pat_fb/joint_timerxl.py --data E1 --arm ctrl --seed $s $X > pat_logs/bisect_${V}_s$s.txt 2>&1
