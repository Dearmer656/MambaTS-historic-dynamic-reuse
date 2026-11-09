#!/bin/bash
#SBATCH --job-name=errval
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=elm72,elm11,elm12,elm13,elm14,elm15,elm21,elm22,elm23,elm24,elm25,elm26,elm74
#SBATCH --output=patchmem/logs/server/err_value/pack_%j.txt
# PAT-315 error-value ridge: TASKS="E:3047 ..." or "E:3047:720" (H, default 192) PAR at a time; existing jsons skipped.
# T / EL keep 5664 + 2H stream origins before n_tr (= 6048 at H192, i.e. a fixed 5665 train origins at every H).
# OUT (default err_value) is the output dir; the H sweep uses OUT=patchmem/logs/server/err_value/hsweep.
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
O=${OUT:-patchmem/logs/server/err_value}
mkdir -p $O
PAR=${PAR:-2}
for x in $TASKS; do
  IFS=: read d s h <<< "$x"; h=${h:-192}
  tag=""; [ $h != 192 ] && tag="_H$h"
  f=$O/${d}_errval${STAGE}${tag}_s${s}
  [ -f $f.json ] && continue
  X=""; { [ $d = T ] || [ $d = EL ]; } && X="--m_tr $((5664 + 2 * h)) --chunk 16"
  while [ $(jobs -r | wc -l) -ge $PAR ]; do sleep 20; done
  ITR_H=$h python -m patchmem.eval_err_value${STAGE} --data $d --seed $s $X --out_dir $O > $f.txt 2>&1 || echo "FAIL $d $s $h" &
done
wait
echo done
