#!/usr/bin/env bash
# Like-for-like throughput, back to back on the same box state: BEFORE's pre-change backends vs r5, same
# six-stream full-recipe harness (100% chin, refined seam), >= 60 s x 2 each.
set -uo pipefail
R=/workspace/MuseTalk-perf300; cd $R; O=$R/docs/fps_comparisons/4070s_400fps_20260928/chin_multistream
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; G="scripts/box_guard.sh run --wait-min 120 --min-avail-gb 10"
$G --label chinms_T_baseline_pair -- $PY scripts/chin_multistream_render.py --backend baseline --streams 6 --out-root $O \
   --label T_baseline_pair --loops 12 --repeats 2 --min-timed-s 60 --compare-accepted > $O/T_baseline_pair.log 2>&1
echo "baseline rc=$? $(grep -a '^SUMMARY' $O/T_baseline_pair.log | grep -o '"aggregate_fps_per_repeat": \[[^]]*\]\|"median_aggregate_fps": [0-9.]*\|"all_clips_match_accepted": [a-z]*' | tr '\n' ' ')"
$G --label chinms_T_srcg50_pair -- $PY scripts/chin_multistream_render.py --backend stagewise16_taesdtrt \
   --flags MUSETALK_UNET_STAGEWISE_CACHE_DIR=models/tensorrt_unet_stagewise_sm89_srcg50 --streams 6 --out-root $O \
   --label T_srcg50_pair --loops 18 --repeats 2 --min-timed-s 60 > $O/T_srcg50_pair.log 2>&1
echo "r5 rc=$? $(grep -a '^SUMMARY' $O/T_srcg50_pair.log | grep -o '"aggregate_fps_per_repeat": \[[^]]*\]\|"median_aggregate_fps": [0-9.]*' | tr '\n' ' ')"
