#!/usr/bin/env bash
# TAESD TRT gates + combined GPU-path bench (plan items 2.1, Phase 2 exit). Each step is its own box_guard lease.
set -uo pipefail
R=/workspace/MuseTalk-perf300; D=$R/docs/fps_comparisons/4070s_300fps_impl_20260928
T=$D/taesd_trt; PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; G="$R/scripts/box_guard.sh run --wait-min 60"
cd "$R"
for step in "$@"; do
  echo "=== step $step $(date -u +%H:%M:%S)"
  case "$step" in
    explore) $G --min-avail-gb 6 --label taesd_explore -- $PY $T/explore_recipe.py > $T/explore_recipe.log 2>&1 ;;
    gate)    $G --min-avail-gb 6 --label taesd_gate -- $PY $T/gate_taesd_trt.py > $T/gate_taesd_trt.log 2>&1 ;;
    speed)   $G --min-avail-gb 6 --label taesd_speed -- $PY $T/speed_taesd_trt.py > $T/speed_taesd_trt.log 2>&1 ;;
    gtrack)  $G --min-avail-gb 6 --label taesd_gtrack -- $PY $T/gtrack_taesd_trt.py > $T/gtrack_taesd_trt.log 2>&1 ;;
    combined16) MUSETALK_UNET_BACKEND=trt_stagewise MUSETALK_UNET_STAGEWISE_BATCH=16 MUSETALK_TAESD_BACKEND=trt MUSETALK_TAESD_TRT_STRICT=1 MUSETALK_TRT_FALLBACK=0 \
          $G --min-avail-gb 6 --label bench_combined_stagewise16_taesdtrt -- $PY scripts/bench_gpu_path.py --batch 16 --seconds 180 --warmup 20 \
          --label stagewise_bs16_taesd_trt --out $D/combined/bench_gpu_path_stagewise16_taesdtrt_180s.json > $D/combined/bench_combined16.log 2>&1 ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  echo "=== step $step rc=$? $(date -u +%H:%M:%S)"
done
