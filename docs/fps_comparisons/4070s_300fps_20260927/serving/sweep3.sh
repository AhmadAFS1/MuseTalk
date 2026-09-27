#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
run() { timeout 200 $PY webrtc_transport_load_v2.py "$@" --duration 15 --warmup 6 > /dev/null 2>&1; }
run --streams 1 --mode live --enc x264fast --gil-probe
run --streams 15 --mode mixed --enc x264default --gil-probe
run --streams 15 --mode mixed --enc x264default --idle-decode-threads 16 --gil-probe
run --streams 15 --mode mixed --enc x264fast --predecoded-idle --gil-probe
run --streams 20 --mode mixed --enc x264fast --predecoded-idle --gil-probe
run --streams 15 --mode idle --enc x264default --idle-decode-threads 16
echo SWEEP3_DONE >> webrtc_transport_load_results.jsonl
