#!/usr/bin/env bash
# Per-input-channel smoothing sweep on blkA_thr_8e-06 (fake quant, no training), then smoothing + learned amax.
set -uo pipefail
R=/workspace/MuseTalk-perf300; cd $R; S=$R/docs/fps_comparisons/4070s_400fps_20260928/int8_study
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; G="scripts/box_guard.sh run --wait-min 90 --min-avail-gb 8"
run() { tag=$1; shift; $G --label smooth_$tag -- $PY scripts/int8_layer_study.py --stage recover --recover-recipe blkA_thr_8e-06 --out $S/study.json --recover-smooth-files 32 "$@" > $S/smooth_$tag.log 2>&1
  echo "$tag rc=$? $(grep -E '^recipe |Error|Traceback' $S/smooth_$tag.log | tail -2)"; }
for a in ${ALPHAS:-0.3 0.5 0.7 0.85}; do run a$a --recover-epochs 0 --recover-params amax --recover-smooth-alpha $a --recover-tag e0; done
