#!/bin/bash
# PAT-307 task 6 baseline: EFC-CET (training-free closed-form error transport, arXiv 2610.22872) on frozen A1, 3 seeds, 3x3 grid.
# usage: sbatch patchmem/slurm/efc_cet.sbatch E|W [ced,row|ced|row]   (runs the smoke test first; feats other than the default -> log suffix _<feats>)
#SBATCH --job-name=pat307_efc
#SBATCH --output=patchmem/logs/server/slurm_%j_efc.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=${1:-E}; F=${2:-ced,row}; SF=$([ "$F" = ced,row ] || echo _${F/,/})
echo "commit $(git rev-parse --short HEAD) data $D"
python -u patchmem/efc_cet.py --data $D --feats $F --out patchmem/logs/server/efc$SF --smoke > patchmem/logs/server/efc_cet_${D}${SF}_smoke.txt 2>&1 || { tail -20 patchmem/logs/server/efc_cet_${D}${SF}_smoke.txt; exit 1; }
grep -E "s30|EFC|version" patchmem/logs/server/efc_cet_${D}${SF}_smoke.txt
python -u patchmem/efc_cet.py --data $D --feats $F --out patchmem/logs/server/efc$SF > patchmem/logs/server/efc_cet_${D}${SF}.txt 2>&1
grep -E "  s30|DONE|Error|Traceback" patchmem/logs/server/efc_cet_${D}${SF}.txt
