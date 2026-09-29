#!/usr/bin/env bash
# Recovery study on blkA_thr_8e-06: epoch-0 equivalence check, then amax / amax+bias / amax+bias+wscale.
set -uo pipefail
R=/workspace/MuseTalk-perf300; cd $R; S=$R/docs/fps_comparisons/4070s_400fps_20260928/int8_study
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; G="scripts/box_guard.sh run --wait-min 90 --min-avail-gb 8"
run() { tag=$1; shift; $G --label recover_$tag -- $PY scripts/int8_layer_study.py --stage recover --recover-recipe blkA_thr_8e-06 --out $S/study.json "$@" > $S/recover_$tag.log 2>&1
  echo "$tag rc=$? $(grep -E '^recipe |Error|Traceback' $S/recover_$tag.log | tail -2)"; }
# run e0 --recover-epochs 0 --recover-params amax --recover-tag e0
run amax --recover-params amax
run amaxbias --recover-params amax,bias
run amaxbiasw --recover-params amax,bias,wscale
