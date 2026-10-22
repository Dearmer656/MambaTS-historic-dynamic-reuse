#!/bin/bash
# PAT-311 frozen official Timer-XL + per-position error feedback fitted on out-of-sample (val) errors.
# usage: sbatch pat_slurm/frozen.sbatch E1|W SEED ARM FIT [RANK] [KEEP]
#SBATCH --job-name=pat311_frz
#SBATCH --output=pat_logs/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/OpenLTM
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=$1; s=$2; ARM=$3; FIT=$4; R=${5:-8}; K=${6:-1}
S=$(grep -a ">>>>>>>start training" pat_logs/repro_${D}_s$s.txt | tail -1 | sed -E 's/.*training : //; s/>+$//')
echo "commit $(git rev-parse --short HEAD) frozen $D seed $s arm $ARM fit $FIT r $R keep $K setting $S"
python -u pat_fb/frozen_tf.py --data $D --setting $S --arm $ARM --fit $FIT --rank $R --keep $K --seed $s > pat_logs/frozen_${D}_${ARM}_fit${FIT}_r${R}_k${K}_s$s.txt 2>&1
