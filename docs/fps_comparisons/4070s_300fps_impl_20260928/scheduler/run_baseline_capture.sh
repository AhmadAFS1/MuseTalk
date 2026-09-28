#!/usr/bin/env bash
# Golden baseline capture (plan item 0.3) with TODAY's scheduler (git HEAD), two processes.
# Process A runs the baseline twice in-process (base_r1, base_r1b) and records lossless
# review clips; process B repeats it in a fresh process (base_r2) for the 2-run gate.
set -uo pipefail
cd /workspace/MuseTalk
OUT=docs/fps_comparisons/4070s_300fps_impl_20260928/scheduler
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
VID=/tmp/claude-0/-workspace/866169db-0af2-4cde-9a98-79a398efeda7/scratchpad/videos
mkdir -p "$VID"
scripts/box_guard.sh run --min-avail-gb 11 --wait-min 480 --label sched_golden_base_A -- \
  $PY scripts/replay_scheduler_exactness.py --mode golden \
  --run base_r1:head --run base_r1b:head \
  --video-dir "$VID" --video-jobs bob_t1,bob_mid,jp_d10 --video-seconds 12 \
  --out $OUT/golden_base_A.json > $OUT/golden_base_A.log 2>&1
echo "A rc=$?" >> $OUT/queue_status.txt
scripts/box_guard.sh run --min-avail-gb 11 --wait-min 480 --label sched_golden_base_B -- \
  $PY scripts/replay_scheduler_exactness.py --mode golden --run base_r2:head \
  --out $OUT/golden_base_B.json > $OUT/golden_base_B.log 2>&1
echo "B rc=$?" >> $OUT/queue_status.txt
