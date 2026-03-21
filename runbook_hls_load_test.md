# HLS Startup And Load Test Runbook

## Purpose

This file documents the current commands for:`

1. starting the MuseTalk API server
2. tuning HLS scheduler behavior
3. running HLS load tests with `load_test.py`
4. checking whether the server is healthy during a run

This runbook is focused on HLS. SSE still exists, but HLS is the main path under active tuning.

## Prerequisites

Activate the Python environment:

```bash
source /content/py310/bin/activate
cd /content/MuseTalk
```

Make sure the avatar already exists:

- `test_avatar` must already be prepared

Make sure the sample audio exists:

- `./data/audio/ai-assistant.mpga`

## Start The API Server

### Default startup

Use this when you want the code defaults:

```bash
python api_server.py --host 0.0.0.0 --port 8000
```

This already includes the current HLS scheduler implementation.

### Tuned startup for HLS experiments

Use this when you want to override scheduler and logging settings:

```bash
HLS_SCHEDULER_MAX_BATCH=8 \
HLS_PREP_WORKERS=2 \
HLS_ENCODE_WORKERS=2 \
HLS_MAX_PENDING_JOBS=16 \
HLS_SERVER_TIMING=true \
HLS_LIVE_STARTUP_SEGMENTS=3 \
HLS_LIVE_PREBUFFER_SECONDS=0.0 \
python api_server.py --host 0.0.0.0 --port 8000
```

### Optional logging controls

By default, per-batch GPU memory lease logs are suppressed.

If you want verbose GPU memory lease logging:

```bash
GPU_MEMORY_LOG_ALLOCATIONS=1 python api_server.py --host 0.0.0.0 --port 8000
```

If you only want to log when GPU lease waits become noticeable, adjust the threshold:

```bash
GPU_MEMORY_WAIT_LOG_THRESHOLD_SECONDS=1.5 python api_server.py --host 0.0.0.0 --port 8000
```

### Optional non-HLS compute gate

This still affects the generic live compute gate used by other streaming paths:

```bash
LIVE_MAX_CONCURRENT_GENERATIONS=2 python api_server.py --host 0.0.0.0 --port 8000
```

For HLS specifically, the more important controls are the `HLS_*` scheduler settings above.

## Quick Health Checks

Check basic health:

```bash
curl http://localhost:8000/health
```

Check overall stats:

```bash
curl http://localhost:8000/stats
```

Check HLS session and scheduler stats:

```bash
curl http://localhost:8000/hls/sessions/stats
```

## Run The HLS Load Test

### Single-stage test

Run a single concurrency level:

```bash
python load_test.py \
  --base-url http://localhost:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 2 \
  --hold-seconds 120
```

### Ramp test

Run multiple concurrency levels in sequence:

```bash
python load_test.py \
  --base-url http://localhost:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --ramp 1,2,3,4,5,6 \
  --hold-seconds 120
```

### Throughput-oriented HLS test

Use this when you want to reduce HLS overhead and test a more forgiving live profile:

```bash
python load_test.py \
  --base-url http://localhost:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 2 \
  --segment-duration 2.0 \
  --musetalk-fps 12 \
  --batch-size 4 \
  --hold-seconds 120
```

### Default CLI knobs

Current `load_test.py` flags:

- `--base-url`
- `--avatar-id`
- `--audio-file`
- `--concurrency`
- `--ramp`
- `--hold-seconds`
- `--segment-duration`
- `--playback-fps`
- `--musetalk-fps`
- `--batch-size`

## Manual HLS Validation

### Create a session

```bash
curl -X POST "http://localhost:8000/hls/sessions/create?avatar_id=test_avatar&playback_fps=30&musetalk_fps=15&batch_size=2&segment_duration=2.0&hls_server_timing=true"
```

### Open the player

Open this in the browser:

```text
http://localhost:8000/hls/player/<session_id>
```

Important:

- the browser player may require a tap before autoplay is allowed

### Start the live stream

```bash
curl -X POST \
  -F "audio_file=@./data/audio/ai-assistant.mpga" \
  "http://localhost:8000/hls/sessions/<session_id>/stream"
```

### Check live status

```bash
curl "http://localhost:8000/hls/sessions/<session_id>/status"
```

### Check the live manifest

```bash
curl "http://localhost:8000/hls/sessions/<session_id>/live.m3u8"
```

If the backend is working, `live.m3u8` should contain `segments/...chunk_....ts` entries after live generation begins.

## How To Read Current Results

### Healthy signs

- `live_ready` arrives quickly
- segments are appended to `live.m3u8`
- `active_requests` returns to zero after the run
- `wall_time_s` is not wildly larger than the media duration

### Warning signs

- very large `avg_time_to_live_ready_s`
- `Segments fetched: 0`
- high wall time for short audio
- player remains on idle even though `live_ready` is true

### Important note about current metrics

Right now, the most reliable signal is:

1. whether the backend finishes cleanly
2. whether `live_ready` becomes true
3. whether the live manifest actually contains chunk entries

`Segments fetched` in the load test can still be misleading if the fetch loop misses or swallows browser-like segment retrieval failures.

## Current Tuning Notes

These are the main HLS tuning controls:

- `segment_duration`
- `musetalk_fps`
- `batch_size`
- `HLS_SCHEDULER_MAX_BATCH`
- `HLS_PREP_WORKERS`
- `HLS_ENCODE_WORKERS`
- `HLS_MAX_PENDING_JOBS`

General rule:

- lower `musetalk_fps` can improve stability but reduce motion smoothness
- higher `segment_duration` can improve stability but increase startup latency
- higher `batch_size` can improve GPU efficiency if the scheduler can keep the GPU busy

## Recommended Workflow

1. Start the server.
2. Verify `/health`.
3. Run a single-stream test first.
4. Check `/stats` and `/hls/sessions/stats` during the run.
5. Then test `--concurrency 2`.
6. Only move higher if `time_to_live_ready` and wall time remain reasonable.

## Restart Reminder

After code changes, restart `api_server.py` before running new tests.
