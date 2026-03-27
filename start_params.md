# Start Params

This file is now the source of truth for the current MuseTalk TRT-stagewise HLS
server path.

The server no longer targets the old `/content/py310` venv by default. The
current working runtime is:

- repo: `/content/MuseTalk`
- venv: `/content/py310_trt_exp`
- VAE backend: `trt_stagewise`
- launcher: [`scripts/run_trt_stagewise_server.sh`](/content/MuseTalk/scripts/run_trt_stagewise_server.sh)
- fresh-node setup: [`scripts/setup_trt_stagewise_server_env.sh`](/content/MuseTalk/scripts/setup_trt_stagewise_server_env.sh)

## Fresh Server Setup

Create the current TRT-stagewise server env from scratch:

```bash
cd /content/MuseTalk
bash scripts/setup_trt_stagewise_server_env.sh --clean
```

What that does:

- installs the required system packages for the current server path
- recreates `/content/py310_trt_exp`
- installs the pinned PyTorch + Torch-TensorRT stack
- installs the pinned HLS/api_server dependencies
- downloads and validates the current model weights
- runs a final import smoke test against `api_server.py`

## Startup Scripts

### Stable Baseline

```bash
cd /content/MuseTalk
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
cd /content/MuseTalk
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

For the stable baseline:

```bash
python load_test.py \
  --base-url http://localhost:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --batch-size 4 \
  --playback-fps 24 \
  --musetalk-fps 12 \
  --hold-seconds 30
```

For the widened throughput branch:

```bash
python load_test.py \
  --base-url http://localhost:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --batch-size 8 \
  --playback-fps 24 \
  --musetalk-fps 12 \
  --hold-seconds 30
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
- NVENC is still not the default because raw ffmpeg NVENC session open fails on
  the current host/runtime; `libx264` remains the reliable baseline

## Related Docs

- [`current_start_param_reference.md`](./current_start_param_reference.md)
- [`current_tensorrt_environment_plan.md`](./current_tensorrt_environment_plan.md)
- [`current_model_backend_findings.md`](./current_model_backend_findings.md)
