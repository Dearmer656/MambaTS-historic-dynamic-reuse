#!/bin/bash
# PAT-311 feedback module on a frozen reproduced Timer-XL. usage: sbatch pat_slurm/fb.sbatch E1|W SEED ARM [RANK] [HC]
#SBATCH --job-name=pat311_fb
#SBATCH --output=pat_logs/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/OpenLTM
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=$1; s=$2; ARM=$3; R=${4:-8}; HC=${5:-2}
S=$(grep -a ">>>>>>>start training" pat_logs/repro_${D}_s$s.txt | tail -1 | sed -E 's/.*training : //; s/>+$//')
echo "commit $(git rev-parse --short HEAD) $D seed $s arm $ARM r $R hc $HC setting $S"
python -u pat_fb/fb_timerxl.py --data $D --setting $S --arm $ARM --rank $R --hc $HC --seed $s > pat_logs/fb_${D}_${ARM}_r${R}_hc${HC}_s$s.txt 2>&1
