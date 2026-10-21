#!/bin/bash
# PAT-307 multi-step residuals (patchmem/r1_msres.py) on ETTh2 P=48, same variable-length recipe as O / T0 (L 384..1440, H 2..7 patches).
# usage: sbatch patchmem/slurm/msres_job.sbatch ARM SEED     ARM: OMS (lead-matched retrieval, = O + multi-step values) | T0MS (TEFL-style g, one origin)
#SBATCH --job-name=pat307_msres
#SBATCH --output=patchmem/logs/server/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
ARM=$1; s=$2
export SEED=$s
case $ARM in
  OMS)  export R1_OBS_MS=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5 DES=R1ObMSK0.5Q8h0s$s ;;
  T0MS) export R1_TEFL=ms R1_TEFL_RANK=64 DES=R1TfmsR64h0s$s ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
echo "commit $(git rev-parse --short HEAD) arm $ARM seed $s"
bash patchmem/r1_etth2.sh > patchmem/logs/server/E_${ARM}_s$s.txt 2>&1
