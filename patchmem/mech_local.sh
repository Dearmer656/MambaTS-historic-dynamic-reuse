#!/bin/bash
# PAT-304 mechanism study queue (eval only, patchmem/mech_eval.py), two lanes = at most 2 GPU jobs.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate mamba_iso
G=${GRID:-1440:336,1440:96,384:96}
f() { python -u patchmem/mech_eval.py --data $1 --seed $2 --grid $G --stride ${3:-1} 2>&1 | grep "^\[mech\|^   \|Error\|error" > patchmem/logs/mech/$1_s$2.txt; }
( f E 3047; f E 3048; f E 3049 ) &
( f T 3047; f W 3047 2 ) &
wait
