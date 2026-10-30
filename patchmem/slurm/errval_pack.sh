#!/bin/bash
#SBATCH --job-name=errval
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=elm72,elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74
#SBATCH --output=patchmem/logs/server/err_value/pack_%j.txt
# PAT-315 error-value ridge: TASKS="E:3047 ..." PAR at a time; Traffic keeps 6048 train origins; existing jsons skipped.
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
O=patchmem/logs/server/err_value
PAR=${PAR:-2}
for x in $TASKS; do
  d=${x%%:*}; s=${x##*:}
  [ -f $O/${d}_errval_s${s}.json ] && continue
  X=""; [ $d = T ] && X="--m_tr 6048 --chunk 16"
  while [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 20; done
  python -m patchmem.eval_err_value --data $d --seed $s $X > $O/${d}_errval_s${s}.txt 2>&1 || echo "FAIL $d $s" &
done
wait
echo done
