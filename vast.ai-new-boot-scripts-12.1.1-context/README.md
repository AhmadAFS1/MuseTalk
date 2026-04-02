# Vast.ai New Boot Scripts 12.1.1 Context

Captured on: 2026-04-02

This note is the handoff context from the server-bootstrap debugging session for
MuseTalk on Vast.ai.

## Goal

Get a fresh Vast.ai node to:

1. create the correct TRT-stagewise MuseTalk venv
2. boot the API server cleanly
3. support avatar preparation in the same venv
4. support normal HLS/session inference in the same venv
5. support `load_test.py` runs from that same venv

Target repo and venv:

- repo: `/workspace/MuseTalk`
- venv: `/workspace/.venvs/musetalk_trt_stagewise`

## Current Canonical Startup Chain

The current startup chain we aligned around is:

1. `setup_musetalk.sh`
2. `scripts/setup_trt_stagewise_server_env.sh`
3. `scripts/setup_trt_experiment_env.sh`
4. `scripts/vast_onstart.sh`
5. `scripts/vast_server_ctl.sh`
6. `scripts/run_trt_stagewise_server.sh`
7. `api_server.py`

Current source-of-truth docs:

- `start_params.md`
- `docs/vast_ai_boot.md`

Known stale doc for this use case:

- `api_calls.md` still references the old `/content/py310` world

## What Was Happening

The original setup path worked for inference better than it worked for avatar
preparation.

Observed failures on the old node included:

- shell heredoc/syntax-looking errors during setup
- `ModuleNotFoundError: No module named 'uvicorn'`
- dependency conflict between `aiortc==1.14.0` and `av==17.0.0`
- later runtime failures for avatar prep:
  - `No module named 'mmpose'`
  - `No module named 'mmcv'`
  - `No module named 'mmcv._ext'`
- partial avatar state:
  - missing `./results/v15/avatars/test_avatar/latents.pt`

## Important Root Cause

The single biggest blocker on the old box was not random missing packages.

It was this mismatch:

- local CUDA toolkit on the machine: `11.8`
- Torch inside the TRT venv: `2.5.1+cu121`

That mattered because full `mmcv` is needed for avatar preparation, and on the
newer TRT stack there was no working prebuilt path for this box, so `mmcv`
fell back to a local build. That build then failed because local CUDA `11.8`
did not match Torch CUDA `12.1`.

So:

- the old `cuda-11.8.0-auto` Vast image could serve inference
- but it was not a reliable path to a true single-venv prep + inference box

## What The Docs Said

The repo had two different histories living at once:

### Older stable app path

`README.md` still describes the older full-app setup:

- Python 3.10
- CUDA 11.7 / 11.8 era
- `torch==2.0.1`
- full MMLab stack always installed:
  - `mmengine`
  - `mmcv==2.0.1`
  - `mmdet==3.1.0`
  - `mmpose==1.1.0`

### Earlier TRT experiment plan

The TRT planning docs explicitly documented a two-venv strategy:

- stable env: `/content/py310`
- TRT experiment env: `/content/py310_trt_exp`

Those same TRT planning docs also explicitly mentioned the newer `cu121`
family:

- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchaudio==2.5.1`
- `torch-tensorrt==2.5.0`

### Current Vast startup docs

The newer Vast docs switched to one main venv path:

- `/workspace/.venvs/musetalk_trt_stagewise`

But they still kept avatar-prep deps optional by default.

That is why the behavior felt contradictory: the server bootstrap had moved to
the newer single-runtime venv, but avatar prep was still not treated as part of
the default autoscaling install.

## Repo Changes Made During This Session

### 1. `scripts/setup_trt_stagewise_server_env.sh`

Fixes:

- moved `require_command "$PYTHON_BIN"` until after optional apt bootstrap
- added post-apt validation for:
  - `python3.10`
  - `ffmpeg`
  - `curl`
  - `git`
- added `--full-stack` mode as a clearer alias for "server deps + avatar-prep
  deps in one venv"

### 2. `scripts/setup_trt_experiment_env.sh`

Fixes:

- `AV_VERSION` changed from `17.0.0` to `16.1.0`
  - reason: `aiortc==1.14.0` requires `av<17.0.0`
- added stronger MMLab install flow:
  - install `openmim`
  - install `setuptools<81`
  - install `ninja`
  - install `psutil`
  - try `mim install mmcv`
  - then try `pip install --no-build-isolation mmcv`
  - then fall back to `mmcv-lite`
- added `--full-stack`
- added fail-fast CUDA/toolkit validation before avatar-prep deps:
  - if local CUDA toolkit does not match Torch CUDA, setup now exits early with
    a clear message instead of half-building the env

### 3. `scripts/vast_onstart.sh`

Fixes:

- strengthened venv validation to catch half-built envs by importing:
  - `api_server`
  - `torch`
  - `torch_tensorrt`
  - `tensorrt`
  - `fastapi`
  - `uvicorn`
  - `aiohttp`
  - `soundfile`
  - `librosa`
  - `imageio`
  - `omegaconf`
  - `ffmpeg`
  - `aiofiles`
  - `av`
  - `multipart`
- validation runs from repo root
- checks `torch.cuda.is_available()`

Note:

- `vast_onstart.sh` still uses `SETUP_INSTALL_AVATAR_PREP_DEPS=1`
- it does not yet expose a dedicated `SETUP_FULL_STACK=1` env var

### 4. Partial avatar handling

Fixes:

- `scripts/api_avatar.py`
  - now detects incomplete avatar folders
  - rebuilds incomplete avatars during prep mode instead of pretending they are
    valid
  - raises a clearer error in inference mode if the avatar is incomplete
- `api_server.py`
  - `/avatars/list` no longer lists incomplete avatars as valid

### 5. MMLab compatibility shim

Added:

- `musetalk/utils/mmlab_compat.py`

Purpose:

- avoids the Torch 2.5 / MMEngine `Adafactor` duplicate-registration issue

Wired into:

- `musetalk/utils/preprocessing.py`
- `scripts/preprocess.py`

## What Was Verified

### Setup path

- shell syntax checks passed on the boot/setup scripts
- `uvicorn` dependency issue was fixed
- the `aiortc` / `av` conflict was fixed

### Server startup

The TRT-stagewise server was confirmed healthy on the old box for inference.

Successful startup eventually showed:

- stagewise TRT warmup completed
- VAE decode backend active: `tensorrt_stagewise`
- models loaded
- audio + Whisper warmup complete
- HLS GPU scheduler started
- `MuseTalk API Server ready`

Health endpoint returned healthy while the server was running.

### Avatar-prep dependency state on old 11.8 box

Not fully solved as a single-venv full-stack path.

Reason:

- full `mmcv` still needed a matching CUDA toolkit
- the local 11.8 toolkit blocked the Torch 2.5.1 + cu121 stack from becoming a
  true full-prep environment

## Important Current Conclusion

For a true single venv that can do:

- avatar preparation
- normal HLS/session inference
- `load_test.py`

the recommended move is:

- create a new Vast template or machine using `cuda-12.1.1-auto`

Do not keep using the older `cuda-11.8.0-auto` image for this goal.

## Recommended New Vast Template

Use:

- image: `vastai/pytorch:cuda-12.1.1-auto`

Avoid for now:

- `cuda-11.8.0-auto`
- `cuda-12.8`
- `cuda-12.9`
- `cuda-13.x`

Reason:

- the repo setup is currently pinned to the `cu121` family

## Recommended Boot Script For The New CUDA 12.1.1 Node

Use this as the on-start script:

```bash
set -euo pipefail

REPO_DIR=/workspace/MuseTalk
REPO_URL="https://github.com/AhmadAFS1/MuseTalk.git"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

SETUP_CLEAN=1 \
SETUP_INSTALL_AVATAR_PREP_DEPS=1 \
STARTUP_TIMEOUT_SECONDS=1800 \
PROFILE=baseline \
PORT=8000 \
bash scripts/vast_onstart.sh
```

Why these flags:

- `SETUP_CLEAN=1`
  - guarantees a clean first build on the fresh box
- `SETUP_INSTALL_AVATAR_PREP_DEPS=1`
  - required for single-venv avatar preparation support
- `STARTUP_TIMEOUT_SECONDS=1800`
  - safer for first boot because install + TRT warmup can take a long time

## First Validation On The New Machine

After the machine comes up, confirm the base image matches the venv intent:

```bash
nvcc --version

/workspace/.venvs/musetalk_trt_stagewise/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
```

Wanted result:

- `nvcc` reports CUDA `12.1`
- Torch reports `2.5.1+cu121`
- `torch.version.cuda` reports `12.1`

## After Setup Finishes

Check server health:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/stats
curl http://127.0.0.1:8000/hls/sessions/stats
```

## Avatar Prep Smoke Test

Example request shape:

```bash
curl -X POST "http://127.0.0.1:8000/avatars/prepare?avatar_id=test_avatar&batch_size=4&bbox_shift=5&force_recreate=true" \
  -F "video_file=@./data/video/chatgpt_moving_vid.mp4"
```

## Load Test Smoke Test

The current `load_test.py` path only needs `aiohttp` and the server to be up.

Example:

```bash
cd /workspace/MuseTalk
source /workspace/.venvs/musetalk_trt_stagewise/bin/activate

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

## Current Open Risk

The scripts are now aligned toward a single-venv full-stack install, but the
final proof still depends on a fresh real run on the new `cuda-12.1.1-auto`
Vast node.

In other words:

- inference path is already validated
- prep path has been reworked for the correct dependency family
- the old 11.8 box was the blocker
- the next machine is the real end-to-end confirmation

## Helpful Reminder

If a future node ever fails again with confusing prep-related import errors,
check these first:

1. `nvcc --version`
2. `python -c "import torch; print(torch.version.cuda)"`
3. whether the boot script actually included:
   - `SETUP_INSTALL_AVATAR_PREP_DEPS=1`

Those three checks should eliminate a lot of wasted time.
