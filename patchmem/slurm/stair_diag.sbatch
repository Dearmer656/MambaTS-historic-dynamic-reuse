#!/bin/bash
# PAT-313 staircase diagnostic (patchmem/stair_diag.py): predictable share of the current error row from row / fresh staircase / CED
# on TEFL checkpoints, L1440 test. Runs D:H in parallel on ONE GPU. usage: sbatch --gres=gpu:TYPE:1 patchmem/slurm/stair_diag.sbatch "E:192 W:192 ..."
#SBATCH --job-name=pat313_stair
#SBATCH --output=patchmem/logs/server/slurm_%j_stair.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude=elm72
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "commit $(git rev-parse --short HEAD) $(hostname) $(date)"
for r in $1; do D=${r%%:*}; H=${r##*:}; python -u patchmem/stair_diag.py --data $D --H $H > patchmem/logs/server/stair/${D}_H$H.txt 2>&1 & sleep 15; done
wait
grep -h "\[stair\]\|Traceback" patchmem/logs/server/stair/*.txt
