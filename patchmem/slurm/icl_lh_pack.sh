#!/bin/bash
# Usage: icl_lh_pack.sh <L> <H> <train 0|1> <eval_parallel>
# PAT-315 horizon sweep on Traffic: (optionally) train iTransformer base at (L, H) for seeds 3047/2024/2025 in
# parallel on one GPU, then run training-free ICL (eval_raw_multistep, eval_raw_ylen) for the 3 seeds.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ITR_L=$1 ITR_H=$2
D=patchmem/logs/server; E=$D/error_path
if [ "$3" = 1 ]; then
  for S in 3047 2024 2025; do python -u -m patchmem.run_itransformer_base --data T --seed $S --seq_len $1 --pred_len $2 > $D/itr_base_T_L$1_H$2_s$S.txt 2>&1 || echo "FAIL train $1 $2 $S" & done; wait
fi
for S in 3047 2024 2025; do echo $S; done | xargs -P "$4" -I{} bash -c "python -u -m patchmem.eval_raw_multistep --seed {} > $E/T_H$2_L$1_phase_multistep_s{}.txt 2>&1 || echo FAIL ms $1 $2 {}; python -u -m patchmem.eval_raw_ylen --seed {} > $E/T_H$2_L$1_phase_rawylen2_s{}.txt 2>&1 || echo FAIL ylen $1 $2 {}"
