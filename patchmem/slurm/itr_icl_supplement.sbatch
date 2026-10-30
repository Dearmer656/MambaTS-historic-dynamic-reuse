#!/bin/bash
# Usage: sbatch this.sbatch E|M1|M2|W|T pure|tefl-full|tefl-periodic
# Three seeds share one A100, except Traffic runs two then one for memory.
#SBATCH --job-name=itr_icl_supp
#SBATCH --output=patchmem/logs/server/itr_icl_supp_%j.out
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00

set -euo pipefail
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p patchmem/logs/server/itr_icl_supp

code=${1:?dataset code}
mode=${2:?pure, tefl-full, or tefl-periodic}
case "$code:$mode" in
  E:pure) cells='576:D1,1008:D1,1440:D1,1440:P24,1440:P168,2880:P24,2880:P168';;
  M1:pure|M2:pure) cells='576:D1,1008:D1,1440:D1,1440:P96,1440:P672,2880:P96,2880:P672';;
  W:pure) cells='576:D1,1440:P144,2880:P144,2880:P1008,4320:P144,4320:P1008,5760:P1008';;
  T:pure) cells='576:D1,1440:P24,1440:P168,2880:P24';;
  E:tefl-full|M1:tefl-full|M2:tefl-full) histories='576,1008,1440';;
  T:tefl-full) histories='576';;
  E:tefl-periodic) histories='1440,2880'; periods='24,168';;
  M1:tefl-periodic|M2:tefl-periodic) histories='1440,2880'; periods='96,672';;
  *) echo "Unsupported $code:$mode" >&2; exit 2;;
esac

chunk=16; forecast_bs=16
if [[ "$code" == W ]]; then chunk=8; fi
if [[ "$code" == T ]]; then chunk=4; forecast_bs=8; fi

run_seed() {
  local seed=$1 out log
  out="patchmem/logs/server/itr_icl_supp/${code}_${mode}_s${seed}.json"
  log="patchmem/logs/server/itr_icl_supp/${code}_${mode}_s${seed}_${SLURM_JOB_ID}.txt"
  case "$mode" in
    pure)
      python -u -m patchmem.eval_itransformer_pure_icl \
        --data "$code" --seed "$seed" --cells "$cells" \
        --clip_w1 0.1 --chunk "$chunk" --forecast_bs "$forecast_bs" \
        --out "$out" > "$log" 2>&1 ;;
    tefl-full)
      python -u -m patchmem.eval_itransformer_tefl_icl \
        --data "$code" --seed "$seed" --histories "$histories" --strides 1 \
        --chunk "$chunk" --forecast_bs "$forecast_bs" \
        --baseline_tag TEFL_paper_L96_E192 --out "$out" > "$log" 2>&1 ;;
    tefl-periodic)
      python -u -m patchmem.eval_itransformer_tefl_icl_periodic \
        --data "$code" --seed "$seed" --histories "$histories" --periods "$periods" \
        --chunk "$chunk" --forecast_bs "$forecast_bs" \
        --baseline_tag TEFL_paper_L96_E192 --out "$out" > "$log" 2>&1 ;;
  esac
}

pids=()
for seed in 3047 2024; do run_seed "$seed" & pids+=("$!"); done
if [[ "$code" != T ]]; then run_seed 2025 & pids+=("$!"); fi
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then exit 1; fi
if [[ "$code" == T ]]; then run_seed 2025; fi
