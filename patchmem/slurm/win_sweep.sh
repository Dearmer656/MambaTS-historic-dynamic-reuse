#!/bin/bash
# PAT-315 retrieval key-window sweep: (1) eval_bwin B variants B24..B720 (ext npz); (2) eval_sr per S_e key window
# (--se_len) with ext B + B variants + loaded P; static val_fit stack.  Traffic L720 H192, 3 seeds in parallel.
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
export PYTHONPATH=/project/nlp-work5/hongyu-s/state_path_attn/_pylib PATH=/cl/work5/hongyu-s/conda/envs/mamba_iso/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 ITR_H=192 ITR_L=720
O=patchmem/logs/server/err_win; BW=patchmem/logs/server/err_bwin
mkdir -p $O $BW
EXT=B,B24@$BW,B48@$BW,B96@$BW,B168@$BW,B336@$BW,B720@$BW
for S in 3047 2024 2025; do
  { [ -f $BW/T_H192_L720_s${S}_ext.npz ] || python -u -m patchmem.eval_bwin --seed $S > $BW/T_H192_L720_s$S.txt 2>&1 || echo "FAIL bwin $S"
    for l in 24 48 96 168 336 720; do
      python -u -m patchmem.eval_sr --data T --seed $S --base shrot --freeze_theta --bptt_h 168 --early_stop --val_split --arms s1_P \
        --load_carry patchmem/logs/server/err_s2 --ext $EXT --ext_dir patchmem/logs/server/err_stack/fc --se_len $l \
        --out_dir $O/se$l > $O/se${l}_s$S.txt 2>&1 || echo "FAIL sr $l $S"
    done; } &
done; wait
