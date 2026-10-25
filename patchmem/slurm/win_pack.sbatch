#!/bin/bash
# PAT-307 task 6: several win_check.py DATA:BASE runs (one --arms) in parallel on ONE GPU (user 2026-10-25: pack tasks per card, any free
# GPU type; override gres on the sbatch line; elm72 excluded: mamba triton fails there). Smoke then full; logs {win_ck|win_attn|win_res}_{D}_{B}.
# usage: sbatch --gres=gpu:TYPE:1 patchmem/slurm/win_pack.sbatch ARMS "E:a1 E:tefl ..."
#SBATCH --job-name=pat307_wpack
#SBATCH --output=patchmem/logs/server/slurm_%j_wpack.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --exclude=elm72
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
A=$1; T=$([ $A = ctrl ] && echo win_ck || ([ $A = attn ] && echo win_attn || echo win_res))
echo "commit $(git rev-parse --short HEAD) arms $A runs $2 on $(hostname)"
run() { D=$1; B=$2; O=patchmem/logs/server/${T}_${D}_${B}; python -u patchmem/win_check.py --data $D --base $B --arms $A --smoke > ${O}_smoke.txt 2>&1 || { echo "SMOKE FAIL $D $B"; tail -5 ${O}_smoke.txt; return; }; python -u patchmem/win_check.py --data $D --base $B --arms $A > ${O}.txt 2>&1; echo "end $D $B $(date)"; grep -E "DONE|Error|Traceback" ${O}.txt | head -3; }
for p in $2; do run ${p%%:*} ${p##*:} & sleep 20; done
wait
