# Start Params

This file is now the source of truth for the current MuseTalk TRT-stagewise HLS
server path, especially for Vast.ai-style deployments.

The server no longer targets the old `/content/py310` venv by default. The
current working runtime is:

- repo: `/workspace/MuseTalk`
- venv: `/workspace/.venvs/musetalk_trt_stagewise`
- VAE backend: `trt_stagewise`
- launcher: [`scripts/run_trt_stagewise_server.sh`](/content/MuseTalk/scripts/run_trt_stagewise_server.sh)
- fresh-node setup: [`scripts/setup_trt_stagewise_server_env.sh`](/content/MuseTalk/scripts/setup_trt_stagewise_server_env.sh)
- Vast on-start wrapper: [`scripts/vast_onstart.sh`](/content/MuseTalk/scripts/vast_onstart.sh)
- Vast server control helper: [`scripts/vast_server_ctl.sh`](/content/MuseTalk/scripts/vast_server_ctl.sh)

The shell launchers are path-neutral now. If the repo lives under
`/workspace/MuseTalk`, their defaults resolve to `/workspace/.venvs/...`. If
the repo lives under `/content/MuseTalk`, they still fall back to `/content`.

## Fresh Server Setup

Create the current TRT-stagewise server env from scratch:

```bash
cd /workspace/MuseTalk
bash scripts/setup_trt_stagewise_server_env.sh --clean
```

What that does:

- installs the required system packages for the current server path
- recreates `/workspace/.venvs/musetalk_trt_stagewise`
- installs the pinned PyTorch + Torch-TensorRT stack
- installs the pinned HLS/api_server dependencies
- includes the current WebRTC runtime deps used by `api_server.py`
- downloads and validates the current model weights
- runs a final import smoke test against `api_server.py`

Important note:

- avatar-preparation deps (`mmpose/mmcv/mmdet/mmengine`) are **not** part of the
  default server bootstrap anymore
- this keeps autoscaled inference workers on the stable inference-only path
- if you need avatar preparation on a specific node, opt in with:

```bash
cd /workspace/MuseTalk
bash scripts/setup_trt_stagewise_server_env.sh --clean --install-avatar-prep-deps
```

or in the Vast wrapper:

```bash
SETUP_INSTALL_AVATAR_PREP_DEPS=1 bash scripts/vast_onstart.sh
```

## Startup Scripts

### Stable Baseline

```bash
cd /workspace/MuseTalk
bash scripts/run_trt_stagewise_server.sh --profile baseline
```

This is the safe TRT-stagewise baseline:

- `HLS_SCHEDULER_MAX_BATCH=4`
- `HLS_SCHEDULER_FIXED_BATCH_SIZES=4`
- `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4`
- `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=4`
- worker pools:
  - `HLS_PREP_WORKERS=8`
  - `HLS_COMPOSE_WORKERS=8`
  - `HLS_ENCODE_WORKERS=8`
- encoder:
  - `HLS_CHUNK_VIDEO_ENCODER=libx264`
  - `HLS_CHUNK_ENCODER_PRESET=ultrafast`
  - `HLS_CHUNK_ENCODER_CRF=28`

### Current Throughput Branch

```bash
cd /workspace/MuseTalk
bash scripts/run_trt_stagewise_server.sh --profile throughput_record
```

This is the widened-batch branch that produced the current best average
throughput at `concurrency=8`:

- `HLS_SCHEDULER_MAX_BATCH=16`
- `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16`
- `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4`
- default warmup:
  - `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16`
- worker pools remain:
  - `HLS_PREP_WORKERS=8`
  - `HLS_COMPOSE_WORKERS=8`
  - `HLS_ENCODE_WORKERS=8`

Important caveat:

- warming `4,8,16` together previously OOM'd on the RTX 3090
- that is why the launcher defaults to warming `8,16` on this profile

## Matching Load Tests

Activate the TRT-stagewise venv first:

```bash
cd /workspace/MuseTalk
source /workspace/.venvs/musetalk_trt_stagewise/bin/activate
```

For the stable baseline at `concurrency=8`:

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

For the widened throughput branch at `concurrency=8`:

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

Optional more realistic staggered arrival test:

```bash
python load_test.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --segment-duration 1.0 \
  --playback-fps 30 \
  --musetalk-fps 15 \
  --batch-size 8 \
  --stagger-seconds 0.5
```

## Current Record

Current best average-throughput result on the widened-batch branch:

- server profile:
  - `throughput_record`
- request `batch_size=8`
- `concurrency=8`
- `avg_time_to_live_ready_s=2.197`
- `avg_segment_interval_s=1.408`
- `max_segment_interval_s=2.525`
- `wall_time_s=26.4`
- `avg_gpu_util_pct=85.21`
- `avg_gpu_memory_used_mb=23922`

Important comparison on the same server branch:

- request `batch_size=4`
- `avg_segment_interval_s=1.423`
- `max_segment_interval_s=2.046`

Practical meaning:

- request `batch_size=8` is the current best average-throughput point
- request `batch_size=4` on the same branch still has the better tail latency

## Notes

- the launcher uses the exact venv python path instead of relying on whichever
  `python` is active in the shell
- this avoids the mixed-venv state seen earlier where `VIRTUAL_ENV` and the
  actual interpreter did not match
- `scripts/vast_server_ctl.sh` starts the server in the background with `nohup`
  and writes logs under `/workspace/logs/musetalk`
- to watch live logs on Vast:

```bash
tail -f /workspace/logs/musetalk/api_server_8000.log
```

- to run in the foreground for debugging:

```bash
cd /workspace/MuseTalk
bash scripts/vast_server_ctl.sh stop
PROFILE=baseline PORT=8000 bash scripts/run_trt_stagewise_server.sh
```

- NVENC is still not the default because raw ffmpeg NVENC session open fails on
  the current host/runtime; `libx264` remains the reliable baseline

## Related Docs

- [`current_start_param_reference.md`](./current_start_param_reference.md)
- [`current_tensorrt_environment_plan.md`](./current_tensorrt_environment_plan.md)
- [`current_model_backend_findings.md`](./current_model_backend_findings.md)
- [`docs/vast_ai_boot.md`](./docs/vast_ai_boot.md)
- [`runbook_hls_load_test.md`](./runbook_hls_load_test.md)
