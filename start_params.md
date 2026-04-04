# Start Params

This file is now the source of truth for the current MuseTalk TRT-stagewise HLS
server path, especially for Vast.ai-style deployments.

The server no longer targets the old `/content/py310` venv by default. The
current working runtime is:

- repo: `/workspace/MuseTalk`
- venv: `/workspace/.venvs/musetalk_trt_stagewise`
- VAE backend: `trt_stagewise`
- launcher: [`scripts/run_trt_stagewise_server.sh`](./scripts/run_trt_stagewise_server.sh)
- fresh-node setup: [`scripts/setup_trt_stagewise_server_env.sh`](./scripts/setup_trt_stagewise_server_env.sh)
- Vast on-start wrapper: [`scripts/vast_onstart.sh`](./scripts/vast_onstart.sh)
- Vast server control helper: [`scripts/vast_server_ctl.sh`](./scripts/vast_server_ctl.sh)

The shell launchers are path-neutral now. If the repo lives under
`/workspace/MuseTalk`, their defaults resolve to `/workspace/.venvs/...`. If
the repo lives under `/content/MuseTalk`, they still fall back to `/content`.

## Validated Live State

The current single-venv path has now been verified end-to-end on a CUDA 12.1
node:

- toolkit: CUDA `12.1`
- GPU class tested live: RTX `3090`
- shared venv:
  - server/runtime imports passed:
    - `api_server`
    - `torch`
    - `torch_tensorrt`
    - `tensorrt`
  - avatar-prep imports passed:
    - `mmcv`
    - `mmcv._ext`
    - `mmengine`
    - `mmdet`
    - `mmpose`
- avatar preparation was re-tested successfully after the S3FD face-detector
  weight was added to bootstrap

Current package-side validation on that node:

- `torch==2.5.1+cu121`
- `torch_tensorrt==2.5.0`
- `tensorrt==10.3.0`
- `mmcv==2.1.0`
- `mmengine==0.10.4`
- `mmdet==3.2.0`
- `mmpose==1.3.1`

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
- if you need a single CUDA 12.1 node to support both avatar preparation and
  TRT inference in the same venv, opt in with:

```bash
cd /workspace/MuseTalk
bash scripts/setup_trt_stagewise_server_env.sh --clean --full-stack
```

or in the Vast wrapper:

```bash
SETUP_FULL_STACK=1 bash scripts/vast_onstart.sh
```

Compatibility note:

- `--install-avatar-prep-deps` and `SETUP_INSTALL_AVATAR_PREP_DEPS=1` still
  work
- `--full-stack` and `SETUP_FULL_STACK=1` are the preferred forms because they
  describe the intended one-venv outcome directly
- full-stack avatar preparation requires full `mmcv`; `mmcv-lite` is not
  enough because the preprocessing path imports `mmcv._ext`
- avatar prep now validates `mmcv._ext`, so a "successful" full-stack install
  really means the compiled MMCV ops are present
- if `scripts/vast_onstart.sh` needs to bootstrap an already-existing target
  venv, it now recreates that venv cleanly instead of attempting an unsupported
  in-place upgrade
- `download_weights.sh` now also stages the S3FD face-detector weight at
  `models/face_detection/s3fd.pth`, so avatar prep should not need an external
  runtime download on the first request

## Startup Scripts

### Stable Baseline

```bash
cd /workspace/MuseTalk
bash scripts/run_trt_stagewise_server.sh --profile baseline
```

This short command is the intended canonical launcher. The script itself sets
the current baseline runtime env vars internally.

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
- if startup logs show values like `compose_workers=10` or `encode_workers=10`
  instead of the documented baseline worker counts, a shell env override is
  still active and the launcher is preserving it
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
- the avatar-prep path still prints several upstream warnings today, including
  `torch.load(..., weights_only=False)` and some MMDetection/MMEngine
  deprecation warnings; those warnings were observed during a successful
  end-to-end avatar preparation and are not currently treated as blockers
- if a failed avatar-prep attempt leaves partial files on disk, retry with
  `force_recreate=true`

## Related Docs

- [`current_start_param_reference.md`](./current_start_param_reference.md)
- [`current_tensorrt_environment_plan.md`](./current_tensorrt_environment_plan.md)
- [`current_model_backend_findings.md`](./current_model_backend_findings.md)
- [`docs/vast_ai_boot.md`](./docs/vast_ai_boot.md)
- [`runbook_hls_load_test.md`](./runbook_hls_load_test.md)
