# V100 WebRTC Load Test - 2026-05-22

## Status

- API server is live locally: `http://127.0.0.1:8000/health`
- Public API is live: `http://67.182.226.160:17948/health`
- Browser test page: `http://67.182.226.160:17948/webrtc/lab`
- Prepared avatar: `test_avatar`
- Report files:
  - `load_test_webrtc_v100_baseline_pytorch_20_20_batch4_ramp1_6.json`
  - `load_test_webrtc_v100_baseline_pytorch_20_20_batch4_ramp1_6_detailed.json`

## Hardware And Runtime

- GPU: `Tesla V100-SXM2-32GB`
- GPU memory: `32768 MB`
- Driver: `570.211.01`
- CPU: `Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz`
- CPU topology: `72` logical CPUs, `2` sockets, `18` cores/socket, `2` threads/core
- Repo revision: `f048778`
- Python env: `/workspace/.venvs/musetalk_trt_stagewise`
- PyTorch: `2.5.1+cu121`
- Torch CUDA: `12.1`

## Server Configuration Tested

This run used the currently live V100 server in a conservative baseline mode:

```bash
PROFILE=baseline \
MUSETALK_TRT_ENABLED=0 \
MUSETALK_VAE_BACKEND=pytorch \
MUSETALK_WARM_RUNTIME=0 \
WEBRTC_H264_ENCODER=libx264 \
WEBRTC_SHARED_GPU_SCHEDULER=1 \
WEBRTC_SYNC_MODE=strict_fifo \
WEBRTC_ADAPTIVE_FPS=0 \
WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0 \
HOST=0.0.0.0 \
PORT=8000 \
VENV_PATH=/workspace/.venvs/musetalk_trt_stagewise \
STARTUP_TIMEOUT_SECONDS=300 \
bash scripts/vast_server_ctl.sh start
```

The HLS/WebRTC scheduler came up with `max_combined_batch_size=4` and
`fixed_batch_sizes=[4]`.

Important caveat: this is not the same as the RTX 3090 `throughput_record` TRT
profile used by the existing reference reports. Earlier in this V100 pass,
stagewise TRT warmup for batch `8` completed in about `244.65s`, but the batch
`16` warmup was not completed before this operational load test. Treat this
document as "current V100 live server capacity", not a pure V100-vs-3090
hardware-only benchmark.

## Avatar Prep

`test_avatar` was prepared successfully from:

```bash
data/video/ai_test_default_moving_vid.mp4
```

Prep request:

```bash
curl -sS -X POST --max-time 1200 \
  'http://127.0.0.1:8000/avatars/prepare?avatar_id=test_avatar&batch_size=4&bbox_shift=5&force_recreate=true' \
  -F 'video_file=@data/video/ai_test_default_moving_vid.mp4'
```

Result:

- HTTP `200`
- elapsed `188.543s`
- stored avatar metadata: `version=v15`, `bbox_shift=0`

For `v15`, the current prep path stores `bbox_shift=0`; the requested shift does
not affect this avatar.

## Load Test Command

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python load_test_webrtc.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --ramp 1,2,3,4,5,6 \
  --hold-seconds 10 \
  --segment-duration 1.0 \
  --playback-fps 20 \
  --musetalk-fps 20 \
  --batch-size 4 \
  --stage-ready-timeout 90 \
  --connection-timeout 30 \
  --completion-timeout 240 \
  --gpu-log-interval 10 \
  --report-path load_test_webrtc_v100_baseline_pytorch_20_20_batch4_ramp1_6.json \
  --detail-report-path load_test_webrtc_v100_baseline_pytorch_20_20_batch4_ramp1_6_detailed.json
```

## V100 Results

Target frame interval for `20 fps` is `0.050s`.

| Streams | Completed | Failed | Avg live-ready | Avg frame interval | Approx receive FPS | Max frame interval | Wall time | Avg GPU util | Peak VRAM | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 0 | 21.612s | 0.051s | 19.6 | 0.062s | 39.8s | 25.36% | 16690 MB | clean after cold start |
| 2 | 2 | 0 | 2.772s | 0.055s | 18.2 | 0.118s | 22.3s | 72.50% | 3846 MB | near realtime, minor jitter |
| 3 | 3 | 0 | 4.036s | 0.091s | 11.0 | 0.451s | 36.4s | 64.89% | 3856 MB | completed, throttled |
| 4 | 4 | 0 | 5.824s | 0.126s | 7.9 | 0.512s | 50.7s | 58.28% | 3866 MB | completed, throttled |
| 5 | 5 | 0 | 6.874s | 0.143s | 7.0 | 0.679s | 58.2s | 69.34% | 3876 MB | completed, throttled |
| 6 | 3 | 3 | 7.693s | 0.172s | 5.8 | 0.820s | 240.2s | 20.61% | 3886 MB | failed, API exited mid-stage |

## Capacity Call

- Smooth `20/20` WebRTC capacity on this V100 baseline profile: `1-2` streams.
- Highest completed stage with zero failures: `5` streams.
- Practical upper bound for "it completes but is visibly throttled": `5` streams.
- Unsupported in this run: `6` streams. The API exited after `3/6` sessions
  completed, then the remaining sessions timed out with connection refused.

The `1`-stream live-ready value includes a cold avatar/runtime load. After that
warmup, the `2`-stream stage reached live in `2.772s` average and maintained an
average receive interval close to the `20 fps` target. At `3+` streams the
effective receive rate drops well below the requested 20 fps.

No cgroup OOM kill was recorded:

```text
oom_kill 0
memory.failcnt 0
```

The server log did not show a Python traceback or CUDA OOM at the exit point.
The failure mode matches an abrupt process exit during high concurrent WebRTC
load, similar to prior RTX 3090 WebRTC stress-test failures.

The API was restarted after the test and is live again.

## RTX 3090 Reference Comparison

Existing automated RTX 3090 WebRTC reference:

- Source report: `load_test_webrtc_report_20_20_4_5_6_8streams_8_16_libx264.json`
- Source notes: `current_webrtc_playback_smoothing_findings.md`
- Profile: `throughput_record`, TRT stagewise `8,16`
- Encoder: `libx264`
- Request shape: `20/20 fps`, request `batch_size=8`
- GPU memory profile: 24 GB RTX 3090, peak near `24.1 GB`

| Streams | V100 current baseline | RTX 3090 TRT reference | Read |
| ---: | --- | --- | --- |
| 4 | 4/4, avg interval 0.126s, live-ready 5.824s | 4/4, avg interval 0.084s, live-ready 10.865s | RTX 3090 had faster steady frame cadence; V100 had faster live-ready in this run |
| 5 | 5/5, avg interval 0.143s, live-ready 6.874s | 5/5, avg interval 0.110s, live-ready 6.217s | RTX 3090 was about 30% faster on average frame cadence |
| 6 | 3/6, avg interval 0.172s before failure | 6/6, avg interval 0.136s | RTX 3090 completed this stage; V100 baseline did not |
| 8 | not tested after 6-stream failure | 4/8, avg interval 0.184s, API exited mid-stage | both profiles fail before clean 8-stream automated WebRTC capacity |

Bottom line: under the live configuration tested here, the V100 is behind the
RTX 3090 TRT reference for sustained WebRTC frame cadence and stable concurrency.
The best current V100 operational call is `2` smooth-ish streams, `5`
completion-only streams, and `6` unsupported. The 3090 reference could complete
`6` streams with `libx264`, but those streams were still throttled and not clean
20 fps realtime.

For an apples-to-apples V100-vs-RTX benchmark, rerun this test after completing
V100 TRT stagewise warmup for `8,16` and use request `batch_size=8`.
