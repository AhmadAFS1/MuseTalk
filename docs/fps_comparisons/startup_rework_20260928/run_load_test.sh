#!/usr/bin/env bash
# WebRTC capacity probe used for the startup-rework A/B (2026-09-28).
# Drives load_test_webrtc.py against a running server and kills the client if
# host MemAvailable falls below a floor, so a probe can never OOM the server.
#
#   run_load_test.sh <label> [ramp] [avatar_id] [base_url]
set -uo pipefail

LABEL=${1:?label}
RAMP=${2:-4,8,12}
AVATAR=${3:-indian_realtime_talking_20f9845543}
BASE_URL=${4:-http://127.0.0.1:8000}
FLOOR_KB=${LOAD_TEST_MEM_FLOOR_KB:-2621440}   # 2.5 GiB
REPO=/workspace/MuseTalk
OUT=$REPO/docs/fps_comparisons/startup_rework_20260928
PY=${LOAD_TEST_PYTHON:-/workspace/.venvs/musetalk_trt_stagewise/bin/python}

cd "$REPO"
python3 "$OUT/gen_fps_sampler.py" sample --url "$BASE_URL" --out "$OUT/${LABEL}_gen.jsonl" &
SAMPLER=$!
"$PY" load_test_webrtc.py \
    --base-url "$BASE_URL" \
    --avatar-id "$AVATAR" \
    --audio-file ./data/audio/ai-assistant.mpga \
    --ramp "$RAMP" \
    --hold-seconds 10 \
    --segment-duration 1.0 \
    --playback-fps 20 \
    --musetalk-fps 20 \
    --batch-size 8 \
    --stage-ready-timeout 120 \
    --connection-timeout 45 \
    --completion-timeout 300 \
    --report-path "$OUT/${LABEL}.json" \
    --detail-report-path "$OUT/${LABEL}_detailed.json" \
    > "$OUT/${LABEL}.log" 2>&1 &
CLIENT=$!

min_kb=999999999
while kill -0 "$CLIENT" 2>/dev/null; do
    avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    (( avail < min_kb )) && min_kb=$avail
    if (( avail < FLOOR_KB )); then
        echo "[watchdog] MemAvailable ${avail} kB < floor ${FLOOR_KB} kB; stopping client" | tee -a "$OUT/${LABEL}.log"
        kill -INT "$CLIENT" 2>/dev/null; sleep 5; kill -9 "$CLIENT" 2>/dev/null
        break
    fi
    sleep 1
done
wait "$CLIENT"; rc=$?
kill -TERM "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
python3 "$OUT/gen_fps_sampler.py" summarize "$OUT/${LABEL}_gen.jsonl" > "$OUT/${LABEL}_gen_summary.json" || true
cat "$OUT/${LABEL}_gen_summary.json" | tee -a "$OUT/${LABEL}.log"
echo "[watchdog] client rc=$rc min_mem_available_gb=$(awk -v k="$min_kb" 'BEGIN{printf "%.2f", k/1048576}')" | tee -a "$OUT/${LABEL}.log"
exit "$rc"
