#!/usr/bin/env bash
# usage: run_int8_set.sh <name> <blocks-to-build> <all-int8-blocks>
set -uo pipefail
R=/workspace/MuseTalk-perf300; cd $R; D=$R/docs/fps_comparisons/4070s_300fps_impl_20260928/unet_fp16
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; C=calibration/unet_multi_avatar_20260928; G="scripts/box_guard.sh run --wait-min 90"
name=$1; build=$2; root=models/tensorrt_unet_stagewise_sm89_$name
$G --min-avail-gb 8 --label build_$name -- $PY scripts/build_unet_stagewise.py --batch 16 --opt-level 5 --root $root --blocks $build --int8-blocks $build --force --report $D/build_${name}_report.json > $D/build_$name.log 2>&1; echo "$name build rc=$?"
$G --min-avail-gb 6 --label chain_bench_$name -- $PY $D/stepB_chain_bench.py --batch 16 --seconds 90 --root $root --label $name --out $D/stepB_chain_bench_bs16_$name.json > $D/stepB_chain_bench_$name.log 2>&1; echo "$name bench rc=$?"
for split in main holdout; do dir=$C; [[ $split == holdout ]] && dir=$C/holdout
  MUSETALK_UNET_BACKEND=trt_stagewise MUSETALK_UNET_STAGEWISE_BATCH=16 MUSETALK_UNET_STAGEWISE_CACHE_DIR=$root MUSETALK_TRT_FALLBACK=0 \
  $G --min-avail-gb 6 --label gunet_${name}_$split -- $PY scripts/validate_unet_backend.py --backend runtime --capture-dir $dir --padded-batch-size 8 --group-captures 2 --limit 0 --warmup 1 --iters 2 --fail-mae 0.01 --fail-max-abs 0.5 --report-path $D/gunet_${name}_$split.json > $D/gunet_${name}_$split.log 2>&1; echo "$name gunet $split rc=$?"; done
