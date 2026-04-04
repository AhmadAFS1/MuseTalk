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

Historical note from the earlier session:

- at capture time, `vast_onstart.sh` still used `SETUP_INSTALL_AVATAR_PREP_DEPS=1`
- the current repo now also exposes `SETUP_FULL_STACK=1` as the clearer
  preferred flag for single-venv prep + inference nodes

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
SETUP_FULL_STACK=1 \
STARTUP_TIMEOUT_SECONDS=1800 \
PROFILE=baseline \
PORT=8000 \
bash scripts/vast_onstart.sh
```

Why these flags:

- `SETUP_CLEAN=1`
  - guarantees a clean first build on the fresh box
- `SETUP_FULL_STACK=1`
  - preferred flag for a single venv that supports avatar prep + inference
- `STARTUP_TIMEOUT_SECONDS=1800`
  - safer for first boot because install + TRT warmup can take a long time

Compatibility note:

- `SETUP_INSTALL_AVATAR_PREP_DEPS=1` still works
- `SETUP_FULL_STACK=1` is now the clearer preferred flag

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
   - `SETUP_FULL_STACK=1`

Those three checks should eliminate a lot of wasted time.

## Addendum: Vast.ai MuseTalk Migration Notes

Captured on: 2026-04-04

This addendum is important because it changes the likely trajectory for fixing
the current prep + inference environment issue. Treat this section as follow-up
migration guidance gathered after the earlier bootstrap debugging notes above.

## Current Source Of Truth

For Vast.ai-style deployments, the current working MuseTalk server path is:

- repo: `/workspace/MuseTalk`
- venv: `/workspace/.venvs/musetalk_trt_stagewise`
- VAE backend: `trt_stagewise`
- fresh-node setup: `scripts/setup_trt_stagewise_server_env.sh`
- foreground launcher: `scripts/run_trt_stagewise_server.sh`
- Vast on-start wrapper: `scripts/vast_onstart.sh`
- server control helper: `scripts/vast_server_ctl.sh`

Important:

- the older root `README.md` install path is the legacy Torch 2.0.1 / CUDA
  11.8 / MMLab flow
- that is not the current TRT-stagewise Vast server bootstrap path
- do not use `scripts/run_api_server.sh` as the main TRT startup path
- the preferred launcher is `scripts/run_trt_stagewise_server.sh`, because it
  pins the correct venv and TRT-related env vars

## Recommended Vast Layout

Use a persistent workspace and keep the repo there:

```bash
/workspace/MuseTalk
```

Default helper paths:

- repo: `/workspace/MuseTalk`
- venv: `/workspace/.venvs/musetalk_trt_stagewise`
- logs: `/workspace/logs/musetalk`

This avoids the older `/content/...` style paths.

## Fresh Server Setup

For a brand-new Vast node:

```bash
cd /workspace/MuseTalk
bash scripts/setup_trt_stagewise_server_env.sh --clean
bash scripts/run_trt_stagewise_server.sh --profile baseline
```

What `setup_trt_stagewise_server_env.sh` does:

- installs required system packages
- creates `/workspace/.venvs/musetalk_trt_stagewise`
- installs the pinned Torch / Torch-TensorRT stack
- installs the pinned API/server runtime deps
- downloads and validates model weights
- runs a final import smoke test against `api_server.py`

Important:

- the stagewise TRT runtime is compiled and warmed in memory at startup
- you do not need to pre-run a separate TensorRT export just to boot the API
  server

## Vast On-Start Script For A Fresh Server

Use this in the Vast.ai on-start field:

```bash
set -euo pipefail

REPO_DIR=/workspace/MuseTalk
REPO_URL=https://github.com/AhmadAFS1/MuseTalk.git
BRANCH=main

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
  cd "$REPO_DIR"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
fi

cd "$REPO_DIR"

PROFILE=baseline \
PORT=8000 \
bash scripts/vast_onstart.sh
```

For the widened profile later:

```bash
set -euo pipefail

cd /workspace/MuseTalk

PROFILE=throughput_record \
PORT=8000 \
bash scripts/vast_onstart.sh
```

## Start Script When The App Is Already Installed

If the repo, weights, and venv are already good:

```bash
set -euo pipefail

cd /workspace/MuseTalk

PROFILE=baseline \
PORT=8000 \
AUTO_SETUP=0 \
bash scripts/vast_onstart.sh
```

Use `AUTO_SETUP=0` only when the existing venv is valid.

If you want to start through the control helper instead:

```bash
cd /workspace/MuseTalk
PROFILE=baseline PORT=8000 bash scripts/vast_server_ctl.sh start
```

If you want foreground logs for debugging:

```bash
cd /workspace/MuseTalk
PROFILE=baseline PORT=8000 bash scripts/run_trt_stagewise_server.sh
```

## Broken Venv Symptoms And Recovery

If you see:

```text
ModuleNotFoundError: No module named 'uvicorn'
```

that means the venv is not a valid server venv. It is usually a partial or
failed setup.

If you see shell syntax errors during setup such as:

```text
bash: syntax error near unexpected token `av.__version__'
bash: syntax error near unexpected token `torch.cuda.is_available'
```

the current repo here suggests the live script copies are fine, so those errors
likely came from one of these:

- a stale or corrupted copy of the script on the remote server
- a partially failed bootstrap
- running the wrong script version on the node

Clean recovery path:

```bash
rm -rf /workspace/.venvs/musetalk_trt_stagewise

cd /workspace/MuseTalk
bash scripts/setup_trt_stagewise_server_env.sh --clean
bash scripts/run_trt_stagewise_server.sh --profile baseline
```

If you want the wrapper to force a rebuild:

```bash
cd /workspace/MuseTalk
PROFILE=baseline PORT=8000 SETUP_CLEAN=1 bash scripts/vast_onstart.sh
```

You usually do not need to delete `models/` unless you want all weights
re-downloaded.

You also do not need to delete every venv on the machine. Delete only the
broken MuseTalk one.

## Common Shell Gotcha On Vast

If you paste a multiline shell block as one flattened line, Bash can throw
errors like:

```text
bash: syntax error near unexpected token `then'
```

That usually means the `if ... then ... fi` block lost its newlines or
semicolons.

Correct multiline form:

```bash
set -euo pipefail

REPO_DIR=/workspace/MuseTalk
REPO_URL="https://github.com/AhmadAFS1/MuseTalk.git"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

PROFILE=baseline \
PORT=8000 \
SETUP_CLEAN=1 \
bash scripts/vast_onstart.sh
```

One-line equivalent:

```bash
set -euo pipefail; REPO_DIR=/workspace/MuseTalk; REPO_URL="https://github.com/AhmadAFS1/MuseTalk.git"; if [ ! -d "$REPO_DIR/.git" ]; then git clone "$REPO_URL" "$REPO_DIR"; fi; cd "$REPO_DIR"; PROFILE=baseline PORT=8000 SETUP_CLEAN=1 bash scripts/vast_onstart.sh
```

## Current TRT Server Profile Details

The TRT-stagewise launcher currently sets these important values:

- `MUSETALK_TRT_ENABLED=1`
- `MUSETALK_VAE_BACKEND=trt_stagewise`
- `MUSETALK_TRT_FALLBACK=0`

Baseline profile defaults:

- `HLS_SCHEDULER_MAX_BATCH=4`
- `HLS_SCHEDULER_FIXED_BATCH_SIZES=4`
- `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=4`

Throughput profile defaults:

- `HLS_SCHEDULER_MAX_BATCH=16`
- `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16`
- `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16`

Important:

- start with `baseline` until the node is stable
- the current reliable default encoder is still `libx264`, not NVENC

## CUDA / Venv Guidance

You do not need different system-wide CUDA installs per venv.

What should be shared across venvs is the host NVIDIA driver.

What can differ is the package stack inside each venv.

Recommended split on one machine:

- `/workspace/.venvs/musetalk_trt_stagewise` for TRT serving
- `/workspace/.venvs/musetalk_avatar_prep` for older avatar-prep / MMLab
  dependencies if needed

Current TRT-stagewise serving venv is pinned around:

- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchaudio==2.5.1`
- `torch-tensorrt==2.5.0`
- `cu121`

Legacy avatar-prep style environment is closer to:

- Python 3.10
- Torch 2.0.1
- CUDA 11.8 / `cu118`
- MMLab packages like `mmcv`, `mmdet`, `mmpose`

Do not try to force both stacks into one venv unless you have already proven
that exact combination works.

## Recommended Vast Base Image

Best first choice for this repo:

- `vastai/pytorch:cuda-12.1-auto`
- or `vastai/pytorch:cuda-12.1.1-auto` if that exact tag is available

Why:

- the current TRT serving setup is pinned to the `cu121` family
- the setup script builds its own venv and installs the pinned stack there
- matching the base image reduces surprises

Avoid as your first serious test:

- `cu128`
- `cu129`
- `cu130`
- `py311`
- `py312`
- `py313`
- `py314`

Those may work later, but they introduce extra variables before the current
server path is stable.

## What The Image Tags Mean

If you see a tag like:

```text
2.5.1-cuda-12.1.1-py310-ipv2
```

it means:

- `2.5.1`: PyTorch version
- `cuda-12.1.1`: CUDA toolkit family in the image
- `py310`: Python 3.10
- `ipv2`: image/platform revision suffix from the image provider

For this repo, `2.5.1` plus `py310` is the best match for the TRT-stagewise
server path.

## Server Management Commands

Once the instance is up:

```bash
cd /workspace/MuseTalk
bash scripts/vast_server_ctl.sh status
bash scripts/vast_server_ctl.sh logs
bash scripts/vast_server_ctl.sh restart
```

To watch logs directly:

```bash
tail -f /workspace/logs/musetalk/api_server_8000.log
```

To verify health:

```bash
curl http://127.0.0.1:8000/health
```

## Practical Recommendation

For tomorrow's retry:

1. use a `cuda-12.1-auto` or `cuda-12.1.1-auto` Vast PyTorch image
2. re-clone the repo if you suspect the remote copy is stale
3. delete only `/workspace/.venvs/musetalk_trt_stagewise` if the server venv is
   broken
4. run `bash scripts/setup_trt_stagewise_server_env.sh --clean`
5. start with `bash scripts/run_trt_stagewise_server.sh --profile baseline`
6. once that is stable, switch to `PROFILE=baseline PORT=8000 bash scripts/vast_onstart.sh`
7. only after baseline is stable, experiment with `throughput_record`

## Bottom Line

The current stable Vast path is:

- build `/workspace/.venvs/musetalk_trt_stagewise`
- launch with `scripts/run_trt_stagewise_server.sh`
- automate with `scripts/vast_onstart.sh`
- manage with `scripts/vast_server_ctl.sh`

If `uvicorn` is missing, the venv is incomplete.

If Bash complains about `then`, the pasted shell block lost its formatting.

If setup emits syntax errors around Python identifiers like `av.__version__`,
suspect a stale script copy or a broken bootstrap and rebuild from a clean venv.
