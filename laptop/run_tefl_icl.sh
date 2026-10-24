#!/bin/bash
# PAT-307 task 6: TEFL compiled into ICL (patchmem/r1_tefl_icl.py, R1TeflICL). TEFL = T0-MS (eps = the full-horizon error of the
# forecast issued one horizon ago); ICL arms replace / correct its W2 by the one-step-GD map of a linear attention over in-window demos
# (error of origin j -> error of origin j+K, all inside the lookback). Order (cheapest first, push after every stage):
#   0 smoke (ETTh2, minutes)
#   1 offline, no training: id0 (phi = identity, eta 1) on A1, error length K = Hmax (id0) or K = the forecast's H (idKH0) |
#     tied0 / hyb0 (W1 / W2 of the T0-MS checkpoint, eta 1)                                                    E, then W id0 / idKH0
#   2 frozen A1 backbone, only the low-rank map trained: TfmsFz (TEFL) | tiedFz (Wq = Wk = W1, ReLU, W2 implicit)          E, W
#   3 joint training (T0-MS recipe): tied | hyb | qk                                                                         E
#   4 Weather: T0-MS (missing baseline), then tied0 / hyb0 on it, then tied | hyb | qk                                       W
# 3 seeds each. A finished run (log has the test audit of 1440/336) is skipped on a rerun, so after an interruption run it again.
# usage (repo root, env active): bash laptop/run_tefl_icl.sh            [stage numbers to run, e.g. "1 2"; default all]
cd "$(dirname "$0")/.."
OUT=patchmem/logs/laptop
mkdir -p $OUT
CKE=./checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue
CKW=./checkpoints/long_term_forecast_weather_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue
SEEDS="3047 3048 3049"
push() { git add -f $OUT/icl6_* && git commit -m "laptop results: tefl_icl $1" && git pull --rebase --autostash origin laptop && git push origin laptop && echo "pushed ($1)"; }
done_() { grep -q "R1Mem audit test\] L=1440 H=336" "$1" 2>/dev/null; }
# run D NAME SEED [env assignments...]: one training / test-only run, log $OUT/icl6_${D}_${NAME}_s${SEED}.txt
run() {
  local D=$1 NAME=$2 s=$3; shift 3
  local LOG=$OUT/icl6_${D}_${NAME}_s$s.txt
  if done_ $LOG; then echo "skip $D $NAME s$s (done)"; return 0; fi
  echo "$D $NAME s$s -> $LOG ($(date))"
  local SH=patchmem/r1_etth2.sh; [ $D = W ] && SH=patchmem/r1_weather.sh
  if ! env SEED=$s R1_TEFL=ms "$@" bash $SH > $LOG 2>&1 || ! done_ $LOG; then
    echo "FAILED $D $NAME s$s, see $LOG"; tail -20 $LOG; push "$D $NAME s$s FAILED"; exit 1
  fi
  grep "R1Mem audit test\] L=1440 H=336" $LOG | cut -c1-200
}
OFF=(R1_ICL_ETA0=1 R1_ICL_ETA_FIX=1 "EXTRA=--is_training 0")                        # test-only, eta fixed at 1
STAGES=${1:-0 1 2 3 4}
echo "commit $(git rev-parse --short HEAD) tefl_icl stages $STAGES ($(date))"
for ST in $STAGES; do case $ST in
0) LOG=$OUT/icl6_smoke.txt
   if ! env SEED=3047 R1_TEFL=ms R1_ICL=hyb EPOCHS=1 R1_MAX_ITERS=30 R1_VAL_STRIDE=32 R1_TEST_L=720,1440 R1_TEST_H=96,336 \
        R1_AUDIT_GRID=1440:336 DES=icl6smoke bash patchmem/r1_etth2.sh > $LOG 2>&1 || ! done_ $LOG; then
     echo "SMOKE FAILED, see $LOG"; tail -20 $LOG; push "smoke FAILED"; exit 1
   fi
   echo "smoke ok" ;;
1) for s in $SEEDS; do
     run E id0 $s R1_ICL=id "${OFF[@]}" R1_EVAL_FROM=${CKE}_R1h0s${s}_0/checkpoint.pth DES=R1IclId0s$s
     run E idKH0 $s R1_ICL=id R1_ICL_KH=1 "${OFF[@]}" R1_EVAL_FROM=${CKE}_R1h0s${s}_0/checkpoint.pth DES=R1IclIdKH0s$s
     run E tied0 $s R1_ICL=tied "${OFF[@]}" R1_EVAL_FROM=${CKE}_R1TfmsR64h0s${s}_0/checkpoint.pth DES=R1IclTied0s$s
     run E hyb0 $s R1_ICL=hyb "${OFF[@]}" R1_EVAL_FROM=${CKE}_R1TfmsR64h0s${s}_0/checkpoint.pth DES=R1IclHyb0s$s
   done
   for s in $SEEDS; do
     run W id0 $s R1_ICL=id "${OFF[@]}" R1_EVAL_FROM=${CKW}_R1Wh0s${s}_0/checkpoint.pth DES=R1WIclId0s$s
     run W idKH0 $s R1_ICL=id R1_ICL_KH=1 "${OFF[@]}" R1_EVAL_FROM=${CKW}_R1Wh0s${s}_0/checkpoint.pth DES=R1WIclIdKH0s$s
   done
   push "stage 1 offline" ;;
2) for D in E W; do
     CK=$CKE; P=R1; [ $D = W ] && CK=$CKW && P=R1W
     for s in $SEEDS; do
       A1=${CK}_${P}h0s${s}_0/checkpoint.pth
       run $D TfmsFz $s R1_FREEZE_FROM=$A1 DES=${P}TfmsFzs$s
       run $D tiedFz $s R1_ICL=tied R1_FREEZE_FROM=$A1 DES=${P}IclTiedFzs$s
     done
     push "stage 2 frozen $D"
   done ;;
3) for v in tied hyb qk; do
     for s in $SEEDS; do run E $v $s R1_ICL=$v DES=R1Icl${v}s$s; done
     push "stage 3 E $v"
   done ;;
4) for s in $SEEDS; do run W Tfms $s DES=R1WTfmsR64h0s$s; done
   push "stage 4 W Tfms"
   for s in $SEEDS; do
     run W tied0 $s R1_ICL=tied "${OFF[@]}" R1_EVAL_FROM=${CKW}_R1WTfmsR64h0s${s}_0/checkpoint.pth DES=R1WIclTied0s$s
     run W hyb0 $s R1_ICL=hyb "${OFF[@]}" R1_EVAL_FROM=${CKW}_R1WTfmsR64h0s${s}_0/checkpoint.pth DES=R1WIclHyb0s$s
   done
   for v in tied hyb qk; do
     for s in $SEEDS; do run W $v $s R1_ICL=$v DES=R1WIcl${v}s$s; done
     push "stage 4 W $v"
   done ;;
esac; done
echo "all done ($(date))"
