#!/bin/bash
# PAT-311 joint Timer-XL + error feedback. usage: sbatch pat_slurm/joint.sbatch E1|W SEED ARM [RANK]   ARM: ctrl|t0r|o|errpath
#SBATCH --job-name=pat311_joint
#SBATCH --output=pat_logs/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/OpenLTM
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=$1; s=$2; ARM=$3; R=${4:-8}
echo "commit $(git rev-parse --short HEAD) joint $D seed $s arm $ARM r $R"
python -u pat_fb/joint_timerxl.py --data $D --arm $ARM --rank $R --seed $s > pat_logs/joint_${D}_${ARM}_r${R}_s$s.txt 2>&1
