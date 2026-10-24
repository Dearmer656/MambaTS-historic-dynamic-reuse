#!/bin/bash
# PAT-307 task 6: TEFL compiled into ICL (patchmem/r1_tefl_icl.py, R1TeflICL). TEFL = T0-MS (eps = the full-horizon error of the
# forecast issued one horizon ago); ICL arms replace / correct its W2 by the one-step-GD map of a linear attention over in-window demos
# (error of origin j -> error of origin j+K, all inside the lookback; n = T - 2K demos, so a short H gives many demos: L1440 H96: 26,
# H192: 22, H336: 16; L720 H96: 11, H336: 1). Trained arms therefore use a FIXED short H (training, val and test all at that H).
# Order (cheapest first, push after every stage / seed):
#   0 smoke (ETTh2, minutes)
#   1 offline, no training, 3x3 grid: id0 (phi = identity, eta 1) on A1, K = Hmax (id0) or K = the forecast's H (idKH0) |
#     tied0 / hyb0 (W1 / W2 of the T0-MS checkpoint, eta 1)                                                    E, then W id0 / idKH0
#   2 ETTh2, H = 96 only (K = 2), 6 arms: a1 (plain R1 at H96) | tefl (T0-MS) | tied (Wq = Wk = W1, ReLU, W2 implicit) |
#     hyb (TEFL W2 + one ICL step) | teflFz / tiedFz (A1 backbone frozen, only the low-rank map trained)
#   3 Weather, H = 96 only, the same 6 arms
#   4 Weather, H = 192 only, the same 6 arms                                  (ETTh2 H = 192 runs on the cluster, job 6144)
# 3 seeds each. A finished run (log has its final test line) is skipped on a rerun, so after an interruption run it again.
# usage (repo root, env active): bash laptop/run_tefl_icl.sh            [stage numbers to run, e.g. "1 2"; default all]
cd "$(dirname "$0")/.."
OUT=patchmem/logs/laptop
mkdir -p $OUT
CKE=./checkpoints/long_term_forecast_ETTh2_r1_MambaTS_ETTh2_ftM_sl1440_ll48_pl336_dm16_nh16_el5_dl2_df2048_fc1_ebtimeF_dtTrue
CKW=./checkpoints/long_term_forecast_weather_r1_MambaTS_custom_ftM_sl1440_ll48_pl336_dm32_nh16_el3_dl2_df512_fc1_ebtimeF_dtTrue
SEEDS="3047 3048 3049"
push() { git add -f $OUT/icl6_* && git commit -m "laptop results: tefl_icl $1" && git pull --rebase --autostash origin laptop && git push origin laptop && echo "pushed ($1)"; }
# a run is complete when its log has the final test line for L=1440 at H = $2 (audit line for TEFL / ICL arms, test line for plain R1)
done_() {
  [ -f "$1" ] || return 1
  if grep -q "^R1Tefl" "$1"; then grep -q "R1Mem audit test\] L=1440 H=$2" "$1"; else grep -q "R1 test\] L=1440 H=$2" "$1"; fi
}
# run D NAME SEED HDONE [env assignments...]: one training / test-only run, log $OUT/icl6_${D}_${NAME}_s${SEED}.txt
run() {
  local D=$1 NAME=$2 s=$3 HD=$4; shift 4
  local LOG=$OUT/icl6_${D}_${NAME}_s$s.txt
  if done_ $LOG $HD; then echo "skip $D $NAME s$s (done)"; return 0; fi
  echo "$D $NAME s$s -> $LOG ($(date))"
  local SH=patchmem/r1_etth2.sh; [ $D = W ] && SH=patchmem/r1_weather.sh
  if ! env SEED=$s "$@" bash $SH > $LOG 2>&1 || ! done_ $LOG $HD; then
    echo "FAILED $D $NAME s$s, see $LOG"; tail -20 $LOG; push "$D $NAME s$s FAILED"; exit 1
  fi
  grep "R1Mem audit test\] L=1440 H=$HD\|R1 test\] L=1440 H=$HD" $LOG | cut -c1-200
}
OFF=(R1_TEFL=ms R1_ICL_ETA0=1 R1_ICL_ETA_FIX=1 "EXTRA=--is_training 0")            # test-only, eta fixed at 1
# fixh D H SEED: the 6 fixed-H arms (H in steps, P = 48: training, val and test all at H)
fixh() {
  local D=$1 H=$2 s=$3 Hp=$(( $2 / 48 )) CK=$CKE P=R1
  [ $D = W ] && CK=$CKW && P=R1W
  local A1=${CK}_${P}h0s${s}_0/checkpoint.pth
  local FX=(R1_HMIN=$Hp R1_HMAX=$Hp R1_TEST_H=$H R1_VAL_GRID=384:$H,720:$H,1440:$H)
  run $D H${H}_a1 $s $H "${FX[@]}" DES=${P}H${H}s$s
  run $D H${H}_tefl $s $H "${FX[@]}" R1_TEFL=ms DES=${P}TfmsH${H}s$s
  run $D H${H}_tied $s $H "${FX[@]}" R1_TEFL=ms R1_ICL=tied DES=${P}IclTiedH${H}s$s
  run $D H${H}_hyb $s $H "${FX[@]}" R1_TEFL=ms R1_ICL=hyb DES=${P}IclHybH${H}s$s
  run $D H${H}_teflFz $s $H "${FX[@]}" R1_TEFL=ms R1_FREEZE_FROM=$A1 DES=${P}TfmsFzH${H}s$s
  run $D H${H}_tiedFz $s $H "${FX[@]}" R1_TEFL=ms R1_ICL=tied R1_FREEZE_FROM=$A1 DES=${P}IclTiedFzH${H}s$s
}
STAGES=${1:-0 1 2 3 4}
echo "commit $(git rev-parse --short HEAD) tefl_icl stages $STAGES ($(date))"
for ST in $STAGES; do case $ST in
0) LOG=$OUT/icl6_smoke.txt
   if ! env SEED=3047 R1_TEFL=ms R1_ICL=tied R1_HMIN=2 R1_HMAX=2 EPOCHS=1 R1_MAX_ITERS=30 R1_VAL_STRIDE=32 R1_TEST_L=720,1440 R1_TEST_H=96 \
        R1_VAL_GRID=720:96,1440:96 R1_AUDIT_GRID=1440:96 DES=icl6smoke bash patchmem/r1_etth2.sh > $LOG 2>&1 || ! done_ $LOG 96; then
     echo "SMOKE FAILED, see $LOG"; tail -20 $LOG; push "smoke FAILED"; exit 1
   fi
   echo "smoke ok" ;;
1) for s in $SEEDS; do
     run E id0 $s 336 R1_ICL=id "${OFF[@]}" R1_EVAL_FROM=${CKE}_R1h0s${s}_0/checkpoint.pth DES=R1IclId0s$s
     run E idKH0 $s 336 R1_ICL=id R1_ICL_KH=1 "${OFF[@]}" R1_EVAL_FROM=${CKE}_R1h0s${s}_0/checkpoint.pth DES=R1IclIdKH0s$s
     run E tied0 $s 336 R1_ICL=tied "${OFF[@]}" R1_EVAL_FROM=${CKE}_R1TfmsR64h0s${s}_0/checkpoint.pth DES=R1IclTied0s$s
     run E hyb0 $s 336 R1_ICL=hyb "${OFF[@]}" R1_EVAL_FROM=${CKE}_R1TfmsR64h0s${s}_0/checkpoint.pth DES=R1IclHyb0s$s
   done
   for s in $SEEDS; do
     run W id0 $s 336 R1_ICL=id "${OFF[@]}" R1_EVAL_FROM=${CKW}_R1Wh0s${s}_0/checkpoint.pth DES=R1WIclId0s$s
     run W idKH0 $s 336 R1_ICL=id R1_ICL_KH=1 "${OFF[@]}" R1_EVAL_FROM=${CKW}_R1Wh0s${s}_0/checkpoint.pth DES=R1WIclIdKH0s$s
   done
   push "stage 1 offline" ;;
2) for s in $SEEDS; do fixh E 96 $s; push "stage 2 E H96 s$s"; done ;;
3) for s in $SEEDS; do fixh W 96 $s; push "stage 3 W H96 s$s"; done ;;
4) for s in $SEEDS; do fixh W 192 $s; push "stage 4 W H192 s$s"; done ;;
esac; done
echo "all done ($(date))"
