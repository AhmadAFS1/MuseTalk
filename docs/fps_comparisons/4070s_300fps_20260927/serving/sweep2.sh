#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
run() { timeout 240 $PY webrtc_transport_load.py "$@" --duration 15 --warmup 6 > /dev/null 2>&1; }
run --streams 15 --mode idle --enc x264default --predecoded-idle
run --streams 20 --mode idle --enc x264fast --predecoded-idle
run --streams 15 --mode mixed --enc x264default --gil-probe
run --streams 15 --mode mixed --enc x264fast --predecoded-idle --gil-probe
run --streams 15 --mode live --enc x264default --gil-probe
run --streams 20 --mode live --enc x264fast --gil-probe
run --streams 20 --mode mixed --enc x264fast --predecoded-idle --gil-probe
run --streams 2 --mode live --enc x264default --gil-probe
echo SWEEP2_DONE >> webrtc_transport_load_results.jsonl
