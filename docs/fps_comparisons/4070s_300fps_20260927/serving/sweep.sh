#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
run() { timeout 240 $PY webrtc_transport_load.py "$@" --duration 15 --warmup 6 > /dev/null 2>&1; }
run --streams 5 --mode idle --enc x264default
run --streams 10 --mode idle --enc x264default
run --streams 15 --mode idle --enc x264default
run --streams 5 --mode live --enc x264default
run --streams 10 --mode live --enc x264default
run --streams 15 --mode live --enc x264default
run --streams 20 --mode live --enc x264default
run --streams 15 --mode live --enc x264fast
run --streams 20 --mode live --enc x264fast
run --streams 12 --mode live --enc nvenc
run --streams 15 --mode idle --enc x264fast
run --streams 20 --mode idle --enc x264fast
echo SWEEP_DONE >> webrtc_transport_load_results.jsonl
