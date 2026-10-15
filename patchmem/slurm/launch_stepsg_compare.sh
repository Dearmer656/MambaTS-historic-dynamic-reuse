#!/bin/bash
# PAT-304: from-scratch backbone + GatedCorrNet, two ways to keep the step loss from pulling the backbone.
#   A (existing, ExpF3GScratchFixS*, jobs 620144-46): step lambda 0.1, backbone+ro lr x0.3
#   B (new,      ExpF3GScratchSgS*)                 : step loss on sg(h-) + C (no backbone gradient), step lambda 1, backbone lr x1
# usage: launch_stepsg_compare.sh smoke   -> GPU unit test, must print ALL PASS first
#        launch_stepsg_compare.sh         -> arm B on ETTh2, 3 seeds
#        launch_stepsg_compare.sh weather -> arm B on Weather too (optional, ~8h/job; Weather A only has 2 seeds)
cd /project/nlp-work5/hongyu-s/state_path_attn/MambaTS-pytorch
B="A0C_CORR=gated,A0C_MODE=cross,A0C_STEP=1,A0C_STEP_SG=1,A0C_SCRATCH_LR=1.0"
if [ "$1" = smoke ]; then sbatch patchmem/slurm/smoke_step_sg.sbatch; exit 0; fi
for s in 3047 3048 3049; do sbatch --export=ALL,SEED=$s,DES=ExpF3GScratchSgS$s,$B patchmem/slurm/a0corr_etth2_h96.sbatch; done
if [ "$1" = weather ]; then for s in 3048 3049; do sbatch --export=ALL,SEED=$s,DES=ExpF3GScratchSgWs$s,$B patchmem/slurm/a0corr_weather_h96.sbatch; done; fi
