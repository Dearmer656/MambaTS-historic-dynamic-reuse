#!/bin/bash
# PAT-307 task 6: the win_attn batch (win_check.py --arms attn) as parallel processes on ONE elm73 6000 GPU (user request 2026-10-25;
# elm73 caps 8 CPUs per GPU -> 1 thread per process). Each pair = DATA BASE; smoke then full, logs as win_check.sbatch.
# usage: sbatch patchmem/slurm/win_attn_elm73.sbatch "E:a1 E:tefl E:a0 W:a1 W:tefl W:a0 T:a1"
#SBATCH --job-name=pat307_wattn73
#SBATCH --output=patchmem/logs/server/slurm_%j_wattn73.txt
#SBATCH --partition=gpu_long
#SBATCH --nodelist=elm73
#SBATCH --gres=gpu:6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "commit $(git rev-parse --short HEAD) pairs $1"
run() { D=$1; B=$2; O=patchmem/logs/server/win_attn_${D}_${B}; python -u patchmem/win_check.py --data $D --base $B --arms attn --smoke > ${O}_smoke.txt 2>&1 || { echo "SMOKE FAIL $D $B"; tail -5 ${O}_smoke.txt; return; }; python -u patchmem/win_check.py --data $D --base $B --arms attn > ${O}.txt 2>&1; echo "end $D $B $(date)"; grep -E "DONE|Error|Traceback" ${O}.txt | head -3; }
for p in $1; do run ${p%%:*} ${p##*:} & sleep 20; done
wait
