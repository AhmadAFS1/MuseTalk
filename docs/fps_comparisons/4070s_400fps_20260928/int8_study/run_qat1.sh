#!/usr/bin/env bash
# Quantization-aware LoRA (merged into FP16 weights) on blkA_thr_8e-06: learning-rate sweep, 12 epochs over calib.
set -uo pipefail
R=/workspace/MuseTalk-perf300; cd $R; S=$R/docs/fps_comparisons/4070s_400fps_20260928/int8_study
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; G="scripts/box_guard.sh run --wait-min 90 --min-avail-gb 8"
run() { tag=$1; shift; $G --label qat_$tag -- $PY scripts/int8_layer_study.py --stage recover --recover-recipe blkA_thr_8e-06 --out $S/study.json "$@" > $S/qat_$tag.log 2>&1
  echo "$tag rc=$? $(grep -E '^recipe |Error|Traceback' $S/qat_$tag.log | tail -2)"; }
# run lora_2e-5 --recover-params amax,lora --recover-lr-amax 0.003 --recover-lr-lora 2e-5 --recover-tag lr2e-5
# run lora_1e-4 --recover-params amax,lora --recover-lr-amax 0.003 --recover-lr-lora 1e-4 --recover-tag lr1e-4
# run bias_2e-5 --recover-params amax,bias --recover-lr-amax 0.003 --recover-lr-bias 2e-5 --recover-tag lr2e-5
run lora_1e-6 --recover-params amax,lora --recover-lr-amax 0.003 --recover-lr-lora 1e-6 --recover-tag lr1e-6
run lora_2e-7 --recover-params amax,lora --recover-lr-amax 0.003 --recover-lr-lora 2e-7 --recover-tag lr2e-7
