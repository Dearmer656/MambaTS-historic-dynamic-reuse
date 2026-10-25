#!/bin/bash
# PAT-307 task 6: win_row (lookback-limited row stream) vs controls (bias / persist / scalar / shuffle / ridge_mean) on base a1 | tefl | a0.
# usage: sbatch patchmem/slurm/win_check.sbatch E|W|T a1|tefl|a0 [ctrl|attn]   (smoke first; logs patchmem/logs/server/{win_ck|win_attn}_{D}_{BASE}[_smoke].txt)
#SBATCH --job-name=pat307_winck
#SBATCH --output=patchmem/logs/server/slurm_%j_winck.txt
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:$PYTHONPATH
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
D=${1:-E}; B=${2:-a1}; A=${3:-ctrl}; O=patchmem/logs/server/$([ $A = attn ] && echo win_attn || echo win_ck)_${D}_${B}
echo "commit $(git rev-parse --short HEAD) data $D base $B arms $A"
python -u patchmem/win_check.py --data $D --base $B --arms $A --smoke > ${O}_smoke.txt 2>&1 || { tail -20 ${O}_smoke.txt; exit 1; }
grep -E "  s30|version|ckpt" ${O}_smoke.txt
python -u patchmem/win_check.py --data $D --base $B --arms $A > ${O}.txt 2>&1
grep -E "  s30|DONE|Error|Traceback" ${O}.txt | cut -c1-300
