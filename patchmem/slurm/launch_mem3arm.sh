#!/bin/bash
# PAT-304 state-level retrieval memory, 3 matched arms (prereg a87e5642, revision 535b5e5f). From scratch, gated CorrNet, cross mode,
# arm-A recipe (step lambda 0.1, backbone+ro lr x0.3) + step_mem lambda 0.1 in every arm. Retrieval heads of 4 channels
# (4 x d_state 16 = 64 features each: ETTh2 8 heads, Weather 16), own softmax per head over that head's channels of C_j. A0C_CKPT must stay unset (from scratch).
#   M0: A0C_MEM=none (m = 0, step_mem on h-)   M1: A0C_MEM=real   M2: A0C_MEM=shuf (key-value pairing rolled one slot within the sequence)
# usage: launch_mem3arm.sh smoke   -> G1-G3 unit tests; must print ALL PASS before anything else
#        launch_mem3arm.sh audit   -> eval-only re-check of the old T1/T2 shuffle control (random-order batches), 6 short jobs
#        launch_mem3arm.sh weather -> 9 runs (primary)      launch_mem3arm.sh etth2 -> 9 runs (secondary)
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
unset A0C_CKPT
COMMON="A0C_CORR=gated,A0C_MODE=cross,A0C_STEP=0.1,A0C_SCRATCH_LR=0.3,A0C_STEP_MEM=0.1,A0C_MEM_GROUP=4"
case "$1" in
  smoke) sbatch patchmem/slurm/smoke_mem_retrieval.sbatch ;;
  audit) for s in 3047 3048 3049; do sbatch --export=ALL,SEED=$s,MODE=mem,DES=ExpMemmems$s patchmem/slurm/mem_weather_h96_shufaudit.sbatch; sbatch --export=ALL,SEED=$s,MODE=shuf,DES=ExpMemshufs$s patchmem/slurm/mem_weather_h96_shufaudit.sbatch; done ;;
  weather) for s in 3047 3048 3049; do for a in 0:none 1:real 2:shuf; do sbatch --time=1-12:00:00 --export=ALL,SEED=$s,DES=ExpMemM${a%%:*}Ws$s,$COMMON,A0C_MEM=${a#*:} patchmem/slurm/a0corr_weather_h96.sbatch; done; done ;;
  etth2) for s in 3047 3048 3049; do for a in 0:none 1:real 2:shuf; do sbatch --export=ALL,SEED=$s,DES=ExpMemM${a%%:*}S$s,$COMMON,A0C_MEM=${a#*:} patchmem/slurm/a0corr_etth2_h96.sbatch; done; done ;;
  *) echo "usage: $0 smoke|audit|weather|etth2"; exit 1 ;;
esac
