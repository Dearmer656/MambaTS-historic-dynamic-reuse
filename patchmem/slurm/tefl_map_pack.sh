#!/bin/bash
#SBATCH --job-name=teflmap
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --exclude=elm72,elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74
#SBATCH --output=patchmem/logs/server/tefl_map/pack_%j.txt
# PAT-315 TEFL map diagnostic: TASKS="E:3047 ..." run PAR at a time on one GPU; existing jsons are skipped.
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
O=patchmem/logs/server/tefl_map
PAR=${PAR:-2}
for x in $TASKS; do
  d=${x%%:*}; s=${x##*:}
  [ -f $O/${d}_tefl_map_s${s}.json ] && continue
  while [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 20; done
  python -m patchmem.diag_tefl_map --data $d --seed $s > $O/${d}_tefl_map_s${s}.txt 2>&1 || echo "FAIL $d $s" &
done
wait
echo done
