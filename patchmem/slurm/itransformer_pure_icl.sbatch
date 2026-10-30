#!/bin/bash
# Usage: sbatch [--dependency=afterok:<base-job>] this.sbatch E|M1|M2|W|T [seed-list] [clip_w1]
#SBATCH --job-name=itr_pure_icl
#SBATCH --output=patchmem/logs/server/itr_pure_icl_%j.out
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=04:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p patchmem/logs/server/itr_pure_icl

dataset=${1:?Specify E, M1, M2, W, or T}
seeds=${2:-3047,2024,2025}
clip=${3:-}
extra=()
suffix=
if [[ -n "$clip" ]]; then
  extra=(--clip_w1 "$clip")
  suffix="_clip${clip//./p}"
fi
case "$dataset" in
  E|M1|M2) cells='1440:D48,2880:D96'; chunk=16; batch=16 ;;
  W) cells='2448:P144,5760:P144'; chunk=16; batch=16 ;;
  T) cells='2880:D48,2880:P168'; chunk=8; batch=8 ;;
  *) exit 2 ;;
esac

IFS=, read -r -a seed_list <<< "$seeds"
pids=()
for seed in "${seed_list[@]}"; do
  python -u -m patchmem.eval_itransformer_pure_icl \
    --data "$dataset" --seed "$seed" --cells "$cells" \
    --chunk "$chunk" --forecast_bs "$batch" \
    "${extra[@]}" \
    --out "patchmem/logs/server/itr_pure_icl/${dataset}_pure_icl${suffix}_s${seed}.json" \
    > "patchmem/logs/server/itr_pure_icl/${dataset}_pure_icl${suffix}_s${seed}_${SLURM_JOB_ID}.txt" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
