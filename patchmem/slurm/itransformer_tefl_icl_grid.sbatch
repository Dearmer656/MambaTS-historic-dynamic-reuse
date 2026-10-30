#!/bin/bash
# Official validation-selected, official test evaluation of in-context W1/W2.
# E, W, and full Traffic share one A100, as requested for the baseline runs.
#SBATCH --job-name=itr_tefl_icl
#SBATCH --output=patchmem/logs/server/itr_tefl_icl/slurm_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p patchmem/logs/server/itr_tefl_icl
echo "start $(date -Is) job=$SLURM_JOB_ID host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

python -u patchmem/eval_itransformer_tefl_icl.py --data E --chunk 64 --forecast_bs 128 \
  > "patchmem/logs/server/itr_tefl_icl/E_s3047_${SLURM_JOB_ID}.txt" 2>&1 &
pid_e=$!
python -u patchmem/eval_itransformer_tefl_icl.py --data W --chunk 32 --forecast_bs 64 \
  > "patchmem/logs/server/itr_tefl_icl/W_s3047_${SLURM_JOB_ID}.txt" 2>&1 &
pid_w=$!
python -u patchmem/eval_itransformer_tefl_icl.py --data T --chunk 8 --forecast_bs 8 \
  > "patchmem/logs/server/itr_tefl_icl/T_s3047_${SLURM_JOB_ID}.txt" 2>&1 &
pid_t=$!

status_e=0
status_w=0
status_t=0
wait "$pid_e" || status_e=$?
wait "$pid_w" || status_w=$?
wait "$pid_t" || status_t=$?
echo "evaluations E=$status_e W=$status_w T=$status_t $(date -Is)"
if (( status_e != 0 || status_w != 0 || status_t != 0 )); then
  exit 1
fi

/home/is/hongyu-s/miniconda3/bin/python patchmem/plot_itransformer_tefl_icl.py
echo "complete $(date -Is)"
