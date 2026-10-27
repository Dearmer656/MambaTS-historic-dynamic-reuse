#!/bin/bash
# Usage: itransformer_longh_icl_pack.sh "<H list>" "<datasets>" <parallel>
# Standalone iTransformer, no TEFL, H in {336,720}: phase average + icl_w2 (phase_controls --backbone base)
# and ICL W2->W1 (pure_icl) on the H=192 best periodic cell; P and the number of demos n are kept from H=192,
# ages start at the first fully observed multiple of P, history = first + (n-1) P + H + L.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib:${PYTHONPATH:-}
export PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=${SLURM_JOB_ID:-local}
declare -A P=([E]=24 [M1]=672 [M2]=96 [W]=144 [T]=168) N=([E]=41 [M1]=3 [M2]=26 [W]=17 [T]=14)
for H in $1; do for D in $2; do
  p=${P[$D]}; n=${N[$D]}; first=$(( (H + p - 1) / p * p )); hist=$(( first + (n - 1) * p + H + 96 ))
  mkdir -p patchmem/logs/server/itr_longh/H$H
  for S in 3047 2024 2025; do
    echo "$H $D $S $hist $p"
  done
done; done | xargs -P "${3:-6}" -L 1 bash -c '
  o=patchmem/logs/server/itr_longh/H$0; tag=${1}_N${3}_P${4}_s${2}
  ITR_H=$0 python -u patchmem/eval_itransformer_tefl_phase_controls.py --backbone base --data $1 --seed $2 --history $3 --period $4 --out $o/${tag}_base.json > $o/${tag}_base_'"$J"'.txt 2>&1 || echo "FAIL phase $0 $1 $2"
  ITR_H=$0 python -u patchmem/eval_itransformer_pure_icl.py --data $1 --seed $2 --cells $3:P$4 --out $o/${tag}_pure.json > $o/${tag}_pure_'"$J"'.txt 2>&1 || echo "FAIL pure $0 $1 $2"'
