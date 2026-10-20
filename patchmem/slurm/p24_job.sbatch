#!/bin/bash
# PAT-307 ETTh2 with patch_len 24 (1 patch = 1 day, weekly period = 7 patches; P=48 has 0.5 / 3.5): does period-aligned patching make
# O's retrieval beat T0? Same recipe as patchmem/r1_etth2.sh except P=24, horizon range 4..14 patches (H 96..336 as before).
# usage: sbatch --time=2-00:00:00 patchmem/slurm/p24_job.sbatch ARM SEED     ARM: A1 | O | TF0 (all errors) | TF0r (hc=4 = last 96 steps)
#SBATCH --job-name=pat307_p24
#SBATCH --output=patchmem/logs/server/slurm_%j.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
ARM=$1; s=$2
export R1_HMIN=4 R1_HMAX=14 SEED=$s EXTRA="--patch_len 24 --stride 24"
case $ARM in
  A1)   export DES=R1P24h0s$s ;;
  O)    export R1_OBS=1 R1_MEM_QK_DIM=8 R1_MEM_SLOT_KEEP=0.5 DES=R1ObK0.5Q8P24h0s$s ;;
  TF0)  export R1_TEFL=native R1_TEFL_HC=0 DES=R1TfnH0R64P24h0s$s ;;
  TF0r) export R1_TEFL=native R1_TEFL_HC=4 DES=R1TfnH4R64P24h0s$s ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
echo "commit $(git rev-parse --short HEAD) arm $ARM seed $s P=24"
bash patchmem/r1_etth2.sh > patchmem/logs/server/E_P24_${ARM}_s$s.txt 2>&1
