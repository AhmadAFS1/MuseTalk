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

## 2026-05-23 Continuation: 32GB V100 Batch-Warmup Experiments

After the baseline run above, the same 32GB V100-class server was retested with
larger TensorRT stagewise warmup buckets and WebRTC `libx264`.

### Known-good recovery profile

This profile reliably starts and was used as the safe fallback after failed
larger-bucket warmups:

```bash
PROFILE=throughput_record \
WEBRTC_H264_ENCODER=libx264 \
HLS_SCHEDULER_MAX_BATCH=16 \
HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16 \
HLS_SCHEDULER_STARTUP_SLICE_SIZE=4 \
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=4,8,16 \
STARTUP_TIMEOUT_SECONDS=2400 \
bash scripts/vast_server_ctl.sh restart
```

Idle VRAM after this profile was roughly `24GB`; scheduler max was `16`.

### Failed larger warmup profiles

`4,8,16,32` failed during batch `32` warmup:

- batch `4`: `178.05s`
- batch `8`: `48.24s`
- batch `16`: `344.51s`
- batch `32`: CUDA OOM
- OOM detail: process was using about `30.86 GiB`; only `894.50 MiB` free;
  TensorRT tried to allocate another `1.00 GiB`; tactic temp allocations also
  showed repeated `~2.16GB` requests.
- Estimate: true batch `32` warmup likely needs roughly `36-40GB` VRAM to be
  reliable on this code path.

`4,8,16,20` failed during batch `20` warmup:

```bash
PROFILE=throughput_record \
HLS_SCHEDULER_MAX_BATCH=20 \
HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,20 \
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=4,8,16,20 \
STARTUP_TIMEOUT_SECONDS=3600 \
bash scripts/vast_server_ctl.sh restart
```

- batch `4`: `50.82s`
- batch `8`: `56.02s`
- batch `16`: `76.68s`
- batch `20`: CUDA OOM
- OOM detail: tried to allocate `640 MiB`; only `552.50 MiB` free; process was
  using about `31.19 GiB`.

`8,16,20` also failed during batch `20` warmup:

```bash
PROFILE=throughput_record \
WEBRTC_H264_ENCODER=libx264 \
HLS_SCHEDULER_MAX_BATCH=20 \
HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16,20 \
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16,20 \
STARTUP_TIMEOUT_SECONDS=3600 \
bash scripts/vast_server_ctl.sh restart
```

- batch `8`: `57.50s`
- batch `16`: `76.97s`
- batch `20`: CUDA OOM
- OOM detail: tried to allocate `1.25 GiB`; only `562.50 MiB` free; process was
  using about `31.18 GiB`.
- Removing the batch `4` bucket did not make batch `20` viable when combined
  with `8,16`.

### Successful batch-24-only profile

The profile that succeeded was intentionally sparse: max batch `24`, with only
the `24` TensorRT bucket warmed and fixed.

```bash
PROFILE=throughput_record \
WEBRTC_H264_ENCODER=libx264 \
HLS_SCHEDULER_MAX_BATCH=24 \
HLS_SCHEDULER_FIXED_BATCH_SIZES=24 \
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=24 \
STARTUP_TIMEOUT_SECONDS=3600 \
bash scripts/vast_server_ctl.sh restart
```

Startup result:

- `Stagewise TRT batch=24 ready in 540.45s`
- health passed after `9m23s`
- idle VRAM immediately after health: about `21.0GB`
- after WebRTC/avatar cache usage: about `25-26GB`
- scheduler reported `max_combined_batch_size=24`

Important behavior: because the only warmed bucket is `24`, WebRTC requests for
smaller batch sizes are resolved upward to batch `24`. For this profile, setting
`batch_size=2` in the wall does not test a true batch-2 TensorRT path; it routes
to the warmed batch-24 path.

### WebRTC load tests on batch-24-only

Automated load tests were run with:

- avatar: `test_avatar`
- encoder: `libx264`
- fps/playback: `20/20`
- requested batch size: `24`
- audio: `./data/audio/ai-assistant.mpga`
- hold seconds: `15`
- segment/chunk duration: `1.0`

Reports:

- `load_test_webrtc_report_v100_20_20_5streams_batch24only_libx264_20260523.json`
- `load_test_webrtc_report_v100_20_20_5streams_batch24only_libx264_20260523_detailed.json`
- `load_test_webrtc_report_v100_20_20_8streams_batch24only_test_avatar_2_libx264_20260523.json`
- `load_test_webrtc_report_v100_20_20_8streams_batch24only_test_avatar_2_libx264_20260523_detailed.json`
- `load_test_webrtc_report_v100_20_20_10streams_batch24only_libx264_20260523.json`
- `load_test_webrtc_report_v100_20_20_10streams_batch24only_libx264_20260523_detailed.json`

| Streams | Completed | Failed | Avg live-ready | Avg frame interval | Approx FPS/stream | Max frame interval | Wall time | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 5 | 0 | 10.226s | 0.088s | 11.36 | 1.874s | 43.8s | 25724 MB |
| 8 | 8 | 0 | 9.230s | 0.165s | 6.06 | 5.640s | 70.9s | 25892 MB |
| 10 | 10 | 0 | 10.037s | 0.191s | 5.24 | 7.639s | 81.5s | 25866 MB |

The 5- and 10-stream automated tests used `test_avatar`; the later 8-stream
test used `test_avatar_2`. All stages completed, but all still showed
frame-throttling warnings. The 8-stream and 10-stream stages had large max frame
intervals (`5.640s` and `7.639s`), so average throughput improved without making
the experience perfectly smooth.

### Comparison against previous V100 `4,8,16` warmup

Previous `4,8,16` WebRTC reports:

- `load_test_webrtc_report_v100_20_20_4_5_6_8streams_4_8_16_libx264_20260523.json`
- `load_test_webrtc_report_v100_20_20_10streams_4_8_16_libx264_20260523.json`

| Streams | `4,8,16` profile | `24 only` profile | Throughput read |
| ---: | --- | --- | --- |
| 5 | 5/5, interval `0.096s`, live-ready `5.403s`, wall `40.2s`, peak `27330 MB` | 5/5, interval `0.088s`, live-ready `10.226s`, wall `43.8s`, peak `25724 MB` | `24 only` improved cadence by about `9.1%`, but live-ready was much slower |
| 8 | 8/8, interval `0.164s`, live-ready `7.208s`, wall `67.7s`, peak `31542 MB` | 8/8, interval `0.165s`, live-ready `9.230s`, wall `70.9s`, peak `25892 MB` | roughly flat cadence (`0.6%` slower), slower live-ready, but about `17.9%` lower peak VRAM |
| 10 | 10/10, interval `0.206s`, live-ready `10.13s`, wall `85.3s`, peak `27428 MB` | 10/10, interval `0.191s`, live-ready `10.037s`, wall `81.5s`, peak `25866 MB` | `24 only` improved cadence by about `7.9%` and wall time by about `4.7%` |

The batch-24-only profile used less peak VRAM during the load tests than the
multi-bucket `4,8,16` profile because it did not keep several warmed TRT
buckets resident.

### RTX 3090 comparison

RTX 3090 reference:

- `load_test_webrtc_report_20_20_4_5_6_8streams_8_16_libx264.json`
- profile: `throughput_record`
- warmed buckets: `8,16`
- request shape: `20/20 fps`, request `batch_size=8`
- GPU memory: 24GB RTX 3090, peak near `24.1GB`

Reference values:

| Streams | RTX 3090 reference |
| ---: | --- |
| 4 | 4/4, avg interval `0.084s`, live-ready `10.865s` |
| 5 | 5/5, avg interval `0.110s`, live-ready `6.217s` |
| 6 | 6/6, avg interval `0.136s`, live-ready `7.056s` |
| 8 | 4/8, avg interval `0.184s`, live-ready `8.979s` |

V100 `4,8,16` vs RTX reference:

- 4 streams: V100 cadence `0.081s` vs RTX `0.084s` -> V100 about `3.7%`
  faster.
- 5 streams: V100 `0.096s` vs RTX `0.110s` -> V100 about `14.6%` faster.
- 6 streams: V100 `0.126s` vs RTX `0.136s` -> V100 about `7.9%` faster.
- 8 streams: V100 completed `8/8` at `0.164s`; RTX completed `4/8` at
  `0.184s`. Cadence was about `12.2%` faster, and completion was materially
  better.

V100 batch-24-only vs RTX:

- 5-stream exact comparison: V100 batch-24-only was about `25.0%` faster on
  average frame interval (`0.088s` vs RTX `0.110s`), but live-ready was slower
  (`10.226s` vs `6.217s`).
- 8-stream exact comparison: V100 batch-24-only completed `8/8` at average
  interval `0.165s`; RTX 3090 completed `4/8` at `0.184s`. Per-stream cadence
  was about `11.5%` faster on the V100 batch-24 run. Aggregate completed FPS was
  about `48.48` vs `21.74`, or `+123.0%`, mostly because the RTX reference only
  completed half the sessions. Live-ready was slightly slower on V100
  (`9.230s` vs `8.979s`), and peak VRAM was higher (`25892 MB` vs `24059 MB`).
- No saved RTX 3090 10-stream report exists. Comparing V100 10-stream
  batch-24-only against RTX 8-stream is not apples-to-apples. The V100 run
  completed `10/10`; the RTX 8-stream reference completed `4/8`.

### Compute-bound vs VRAM-bound read

During the batch-24-only WebRTC runs, VRAM stayed around `25-26GB` rather than
climbing to the full `32GB`. This is expected:

- The server warmed only one TensorRT bucket: `24`.
- `HLS_SCHEDULER_MAX_BATCH=24` prevents the scheduler from intentionally
  combining work above total batch `24`.
- Ten WebRTC streams do not create ten independent model copies; they share the
  runtime and are batched through the scheduler.
- VRAM is allocated for resident model/runtime state, warmed TRT engines, avatar
  cache, buffers, and active work. The GPU does not fill all available VRAM just
  because it exists.
- GPU utilization often reached high or near-100% values while VRAM remained
  below capacity, which points to compute/scheduling/encode/composition limits
  more than raw VRAM capacity.

Current practical interpretation: batch `24` is a useful throughput profile on
the 32GB card, but extra free VRAM does not automatically translate to higher
WebRTC throughput unless the scheduler/runtime can exploit a larger warmed
bucket, additional buckets, or additional replicas. Batch `32` was not viable
on this 32GB GPU with the current warmup path.

### Avatar state and wall URLs

The original load tests above used `test_avatar`. A newer avatar was prepared
afterward:

- avatar id: `test_avatar_2`
- reference video: `data/video/chatgpt_moving_vid.mp4`
- uploaded copy: `uploads/videos/test_avatar_2_chatgpt_moving_vid.mp4`
- prepared files:
  - `results/v15/avatars/test_avatar_2/input_video.mp4`
  - `results/v15/avatars/test_avatar_2/avator_info.json`
  - `results/v15/avatars/test_avatar_2/coords.pkl`
  - `results/v15/avatars/test_avatar_2/mask_coords.pkl`
  - `results/v15/avatars/test_avatar_2/latents.pt`

Preparation command used:

```bash
curl -sS -X POST \
  "http://127.0.0.1:8000/avatars/prepare?avatar_id=test_avatar_2&bbox_shift=0&batch_size=24&force_recreate=true" \
  -F "video_file=@data/video/chatgpt_moving_vid.mp4;filename=chatgpt_moving_vid.mp4"
```

Operational caveat: `/avatars/prepare` is blocking in this server path. Running
it against the live API blocked normal health/status responses while face and
mask extraction was in progress; existing WebRTC wall iframes showed
`Connection lost`. Prepare avatars before interactive wall testing, or move this
work off the live wall server.

Useful wall/API URLs:

```text
WebRTC wall: http://127.0.0.1:8000/webrtc/wall
HLS wall:    http://127.0.0.1:8000/hls/wall
```

Fresh WebRTC group for visual testing:

```bash
curl -sS -X POST \
  "http://127.0.0.1:8000/webrtc/groups/create?avatar_id=test_avatar_2&count=5&fps=20&playback_fps=20&batch_size=24&chunk_duration=1"
```

Fresh HLS group for visual testing:

```bash
curl -sS -X POST \
  "http://127.0.0.1:8000/hls/groups/create?avatar_id=test_avatar_2&count=5&fps=20&playback_fps=20&batch_size=24&segment_duration=1"
```

For this batch-24-only server, use wall batch size `24`. Setting batch size `2`
will be resolved upward to `24` because that is the only warmed bucket.
