#!/bin/bash
# PAT-307 task 6: gain-source diagnostics (patchmem/gain_src.py), several DATA:BASE runs in parallel on ONE GPU (user 2026-10-25: pack tasks
# per card, any free GPU type; override gres / cpus / mem on the sbatch line). Smoke then full per run; logs gain_src_{D}_{B}[_smoke].txt.
# usage: sbatch --gres=gpu:TYPE:1 patchmem/slurm/gain_src_pack.sbatch "E:a1 E:tefl ..."
#SBATCH --job-name=pat307_gsrc
#SBATCH --output=patchmem/logs/server/slurm_%j_gsrc.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "commit $(git rev-parse --short HEAD) runs $1 on $(hostname) $CUDA_VISIBLE_DEVICES"
run() { D=$1; B=$2; O=patchmem/logs/server/gain_src_${D}_${B}; python -u patchmem/gain_src.py --data $D --base $B --smoke > ${O}_smoke.txt 2>&1 || { echo "SMOKE FAIL $D $B"; tail -5 ${O}_smoke.txt; return; }; python -u patchmem/gain_src.py --data $D --base $B > ${O}.txt 2>&1; echo "end $D $B $(date)"; grep -E "DONE|Error|Traceback" ${O}.txt | head -3; }
for p in $1; do run ${p%%:*} ${p##*:} & sleep 20; done
wait
