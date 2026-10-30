#!/bin/bash
# Periodicity vs ICL controls on identical periodic demos (Traffic N2880 P168, 3 seeds).
#SBATCH --job-name=itr_phase_ctrl
#SBATCH --output=patchmem/logs/server/itr_phase_ctrl_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --exclude=elm72

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

D=${1:-T}; N=${2:-2880}; P=${3:-168}; B=${4:-tefl}
mkdir -p patchmem/logs/server/itr_phase_controls
for S in 3047 2024 2025; do
  python -u patchmem/eval_itransformer_tefl_phase_controls.py --data $D --seed $S --history $N --period $P --backbone $B \
    > "patchmem/logs/server/itr_phase_controls/${D}_N${N}_P${P}_s${S}_${B}_${SLURM_JOB_ID}.txt" 2>&1
done
