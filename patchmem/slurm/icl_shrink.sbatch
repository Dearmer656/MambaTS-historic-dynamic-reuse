#!/bin/bash
# PAT-307 task 6 shrinkage (fixed before results, all reported, none selected): offline, ETTh2, 3x3 test grid, eta 1, one seed per job.
#   ridge lam path on A1 (K = H): R1_ICL_RLAM = s in {1, 10, 100, 1000} (lam = s * mean |k|^2)      -> E_iclridgeR{s}KH0_s{SEED}
#   hyb + n-shrink on the T0-MS (TEFL) checkpoint (K = 7): eta n / (n + 8)                           -> E_iclhybN8K7_0_s{SEED}
#   hybridge (ridge anchored at TEFL W2) on the T0-MS checkpoint (K = 7): s in {1, 10, 100, 1000}      -> E_iclhybridgeR{s}K7_0_s{SEED}
# usage: sbatch patchmem/slurm/icl_shrink.sbatch SEED
#SBATCH --job-name=pat307_shrink
#SBATCH --output=patchmem/logs/server/slurm_%j_shrink.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
s=${1:-3047}
CK=./checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue
A1=${CK}_R1h0s${s}_0/checkpoint.pth; TF=${CK}_R1TfmsR64h0s${s}_0/checkpoint.pth
echo "commit $(git rev-parse --short HEAD) seed $s"
# one NAME CKPT KH [env...]
one() {
  local N=$1 C=$2 KH=$3; shift 3
  local LOG=patchmem/logs/server/E_icl${N}_s$s.txt
  env SEED=$s R1_TEFL=ms R1_ICL_KH=$KH R1_ICL_ETA0=1 R1_ICL_ETA_FIX=1 R1_EVAL_FROM=$C DES=R1Icl${N}s$s EXTRA="--is_training 0" "$@" bash patchmem/r1_etth2.sh > $LOG 2>&1
  echo "== $N"; grep -E "R1TeflICL|audit test\] L=(1440| 720)|Traceback|Error" $LOG | cut -c1-300
}
for r in 1 10 100 1000; do one ridgeR${r}KH0 $A1 1 R1_ICL=ridge R1_ICL_RLAM=$r; done
one hybN8K7_0 $TF 0 R1_ICL=hyb R1_ICL_NSHRINK=8
for r in 1 10 100 1000; do one hybridgeR${r}K7_0 $TF 0 R1_ICL=hybridge R1_ICL_RLAM=$r; done
