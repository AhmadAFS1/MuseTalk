# HLS Startup And Load Test Runbook

## Purpose

This file documents the current commands for:

1. starting the MuseTalk API server
2. tuning HLS scheduler behavior
3. running HLS load tests with `load_test.py`
4. checking whether the server is healthy during a run

This runbook is focused on HLS. SSE still exists, but HLS is the main path under active tuning.

## Prerequisites

Activate the Python environment:

```bash
source /workspace/.venvs/musetalk_trt_stagewise/bin/activate
cd /workspace/MuseTalk
```

Make sure the avatar already exists:

- `test_avatar` must already be prepared

Make sure the sample audio exists:

- `./data/audio/ai-assistant.mpga`

## Start The API Server

### Default startup

Use this when you want the current Vast-style background server control:

```bash
cd /workspace/MuseTalk
bash scripts/vast_server_ctl.sh start
```

This already includes the current HLS scheduler implementation and writes logs to
`/workspace/logs/musetalk/api_server_8000.log`.

If you want live logs:

```bash
tail -f /workspace/logs/musetalk/api_server_8000.log
```

If you want the server in the foreground for debugging:

```bash
cd /workspace/MuseTalk
bash scripts/vast_server_ctl.sh stop
PROFILE=baseline PORT=8000 bash scripts/run_trt_stagewise_server.sh
```

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

Run the current common `concurrency=8` baseline comparison:

```bash
python load_test.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --segment-duration 1.0 \
  --playback-fps 30 \
  --musetalk-fps 15 \
  --batch-size 4
```

For the widened throughput branch request shape:

```bash
python load_test.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --segment-duration 1.0 \
  --playback-fps 30 \
  --musetalk-fps 15 \
  --batch-size 8
```

Recent hosted output from May 10, 2026, on the active `throughput_record`
server:

```json
{
  "concurrency": 8,
  "completed": 8,
  "failed": 0,
  "avg_time_to_live_ready_s": 2.265,
  "avg_segment_interval_s": 1.779,
  "max_segment_interval_s": 2.546,
  "wall_time_s": 33.2,
  "avg_gpu_util_pct": 83.59,
  "peak_gpu_util_pct": 100.0,
  "peak_gpu_memory_used_mb": 23984.0
}
```

This validates that the hosted worker can serve eight simultaneous HLS streams
with average cadence below `2.0s`. The run still emits the load-test throttling
warning because the max segment interval exceeded `2.0s`, so tail jitter remains
the edge to watch.

### 30/30 FPS validation

Use this when you want to test full-rate MuseTalk generation instead of the
normal `15/30` shape:

```bash
python load_test.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --segment-duration 1.0 \
  --playback-fps 30 \
  --musetalk-fps 30 \
  --batch-size 8
```

Recent May 10, 2026 results:

```json
{
  "single_stream_4_8": {
    "concurrency": 1,
    "completed": 1,
    "failed": 0,
    "avg_time_to_live_ready_s": 1.513,
    "avg_segment_interval_s": 0.476,
    "max_segment_interval_s": 0.512,
    "wall_time_s": 9.2,
    "peak_gpu_memory_used_mb": 13856.0
  },
  "eight_streams_4_8": {
    "concurrency": 8,
    "completed": 8,
    "failed": 0,
    "avg_time_to_live_ready_s": 3.270,
    "avg_segment_interval_s": 3.826,
    "max_segment_interval_s": 5.574,
    "wall_time_s": 69.0,
    "peak_gpu_memory_used_mb": 13856.0
  },
  "eight_streams_8_16": {
    "concurrency": 8,
    "completed": 8,
    "failed": 0,
    "avg_time_to_live_ready_s": 5.032,
    "avg_segment_interval_s": 3.567,
    "max_segment_interval_s": 6.088,
    "wall_time_s": 66.6,
    "peak_gpu_memory_used_mb": 23920.0
  }
}
```

The `8,16` profile gave a small average-throughput win for `30/30`
(`3.826s -> 3.567s` average segment interval), but worsened startup and tail
latency. Both 8-stream runs are well above the `2.0s` throttle threshold, so
`30/30` should be treated as a low-concurrency quality experiment rather than
the current hosted 8-stream target.

### 24/24 FPS 3-stream validation

Use this when you want to test full-rate 24 FPS generation at lower concurrency:

```bash
python load_test.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 3 \
  --segment-duration 1.0 \
  --playback-fps 24 \
  --musetalk-fps 24 \
  --batch-size 8
```

Recent May 10, 2026 result on the current `8,16` throughput profile:

```json
{
  "concurrency": 3,
  "completed": 3,
  "failed": 0,
  "avg_time_to_live_ready_s": 1.848,
  "avg_segment_interval_s": 1.060,
  "max_segment_interval_s": 1.527,
  "wall_time_s": 19.9,
  "avg_gpu_util_pct": 82.33,
  "peak_gpu_util_pct": 100.0,
  "peak_gpu_memory_used_mb": 23922.0
}
```

This passed cleanly against the `2.0s` throttle threshold. Treat it as a
validated 3-stream quality profile, not as evidence that `24/24` can sustain
the 8-stream hosted target.

### 20/20 FPS 4-5 stream validation

Use this when you want to test full-rate 20 FPS generation around the current
low-concurrency edge:

```bash
python load_test.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 4 \
  --segment-duration 1.0 \
  --playback-fps 20 \
  --musetalk-fps 20 \
  --batch-size 8
```

Recent May 10, 2026 results on the current `8,16` throughput profile:

```json
{
  "five_streams_20_20": {
    "concurrency": 5,
    "completed": 5,
    "failed": 0,
    "avg_time_to_live_ready_s": 2.014,
    "avg_segment_interval_s": 1.477,
    "max_segment_interval_s": 2.550,
    "wall_time_s": 27.5,
    "avg_gpu_util_pct": 78.07,
    "peak_gpu_util_pct": 100.0,
    "peak_gpu_memory_used_mb": 23922.0
  },
  "four_streams_20_20": {
    "concurrency": 4,
    "completed": 4,
    "failed": 0,
    "avg_time_to_live_ready_s": 1.889,
    "avg_segment_interval_s": 1.188,
    "max_segment_interval_s": 2.041,
    "wall_time_s": 22.4,
    "avg_gpu_util_pct": 77.04,
    "peak_gpu_util_pct": 100.0,
    "peak_gpu_memory_used_mb": 23922.0
  }
}
```

Both runs completed, but both emitted the strict load-test throttling warning.
The 5-stream run exceeded the `2.0s` tail threshold by `0.550s`; the 4-stream
run exceeded it by only `0.041s`. Treat `20/20` at four streams as near the
current burst-start edge, not as a completely clean no-warning profile.

### Ramp test

Run multiple concurrency levels in sequence:

```bash
python load_test.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --ramp 1,2,3,4,5,6 \
  --hold-seconds 120
```

### Throughput-oriented HLS test

Use this when you want to reduce HLS overhead and test a more forgiving live profile:

```bash
python load_test.py \
  --base-url http://127.0.0.1:8000 \
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
- `--stagger-seconds`
- `--gpu-index`
- `--gpu-sample-interval`
- `--gpu-log-interval`

## Manual HLS Validation

### Create a session

```bash
curl -X POST "http://localhost:8000/hls/sessions/create?avatar_id=test_avatar&playback_fps=30&musetalk_fps=15&batch_size=2&segment_duration=2.0&hls_server_timing=true"
```

If this returns `404`, the process on port `8000` is not the expected HLS
server path and you should inspect the server log file before trusting load test
results.

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
