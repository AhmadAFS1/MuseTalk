# Vast.ai Boot Flow

This repo already has a current canonical setup path and launch path:

- setup: `setup_musetalk.sh`
- launch: `scripts/run_trt_stagewise_server.sh`

The Vast.ai helpers added here wrap those entrypoints so autoscaled instances
can bring themselves up without manual SSH steps.

## Files

- `scripts/vast_onstart.sh`
  - idempotent Vast instance boot wrapper
  - optionally runs setup if the venv / model files are missing
  - starts the MuseTalk server and waits for `/health`
- `scripts/vast_server_ctl.sh`
  - `start|stop|restart|status|logs` helper for the TRT-stagewise server
  - writes logs and a PID file under `/workspace/logs/musetalk` by default

## Recommended Vast.ai Layout

Use a persistent workspace and keep the repo there.

Example repo path:

```bash
/workspace/MuseTalk
```

Default Vast helper paths:

- repo: `/workspace/MuseTalk`
- venv: `/workspace/.venvs/musetalk_trt_stagewise`
- logs: `/workspace/logs/musetalk`

This avoids hard-coding the Colab-style `/content/...` paths on Vast.

Recommended image for the single-venv prep + inference path:

- `vastai/pytorch:cuda-12.1.1-auto`

## Vast.ai On-Start Script

For a fresh server, paste this into the Vast.ai on-start command field:

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

For the widened throughput profile:

```bash
set -euo pipefail

cd /workspace/MuseTalk
PROFILE=throughput_record \
PORT=8000 \
bash scripts/vast_onstart.sh
```

For a CUDA 12.1.1 node that must support both avatar preparation and TRT
inference from the same venv, use:

```bash
set -euo pipefail

cd /workspace/MuseTalk
SETUP_CLEAN=1 \
SETUP_FULL_STACK=1 \
STARTUP_TIMEOUT_SECONDS=1800 \
PROFILE=baseline \
PORT=8000 \
bash scripts/vast_onstart.sh
```

## Behavior

On boot, `scripts/vast_onstart.sh` will:

1. check whether the TRT-stagewise venv and key model files already exist
2. if missing, run `setup_musetalk.sh` with the configured `VENV_PATH`
3. call `scripts/vast_server_ctl.sh start`
4. wait until `http://127.0.0.1:$PORT/health` responds

If the server is already running, it does not start a duplicate process.

Important current behavior:

- the default autoscaling path is inference-only
- the default setup installs the server runtime deps, including `aiortc`
- avatar-preparation deps (`mmpose/mmcv/mmdet/mmengine`) are optional and are
  only installed when explicitly requested
- `SETUP_FULL_STACK=1` is the preferred flag for a single-vm, single-venv node
  that must handle both avatar prep and inference
- `SETUP_INSTALL_AVATAR_PREP_DEPS=1` is still accepted as a compatibility alias
- if bootstrap is needed and the target venv already exists, the on-start
  wrapper now recreates that venv cleanly instead of attempting an unsupported
  in-place upgrade

## Useful Environment Overrides

All of these can be set in Vast.ai environment variables or inline in the
on-start command:

- `PROFILE=baseline|throughput_record`
- `PORT=8000`
- `HOST=0.0.0.0`
- `REPO_ROOT=/workspace/MuseTalk`
- `VENV_PATH=/workspace/.venvs/musetalk_trt_stagewise`
- `AUTO_SETUP=1`
- `SETUP_CLEAN=0`
- `SETUP_SKIP_APT=auto`
- `SETUP_SKIP_WEIGHTS=0`
- `SETUP_FULL_STACK=0`
- `SETUP_INSTALL_AVATAR_PREP_DEPS=0`
- `STARTUP_TIMEOUT_SECONDS=600`

`SETUP_SKIP_APT=auto` means:

- if the script is running as root, it allows apt installation on first boot
- if it is not running as root, it automatically falls back to `--skip-apt`

## Manual Control Over SSH

Once the instance is up, you can manage the server with:

```bash
cd /workspace/MuseTalk
bash scripts/vast_server_ctl.sh status
bash scripts/vast_server_ctl.sh logs
bash scripts/vast_server_ctl.sh restart
```

To watch live server logs continuously:

```bash
tail -f /workspace/logs/musetalk/api_server_8000.log
```

If you want foreground logs instead of the background `nohup` path:

```bash
cd /workspace/MuseTalk
bash scripts/vast_server_ctl.sh stop
PROFILE=baseline PORT=8000 bash scripts/run_trt_stagewise_server.sh
```

Background logging is expected because `scripts/vast_server_ctl.sh` starts the
server with `nohup` and redirects stdout/stderr into the log file.

## Single-Venv Prep + Inference Nodes

If a node needs to prepare avatars as well as serve inference from the same
CUDA 12.1.1 venv, opt in to the heavier preprocessing stack:

```bash
cd /workspace/MuseTalk
PROFILE=baseline \
PORT=8000 \
SETUP_CLEAN=1 \
SETUP_FULL_STACK=1 \
bash scripts/vast_onstart.sh
```

Compatibility note:

- `SETUP_INSTALL_AVATAR_PREP_DEPS=1` still works
- `SETUP_FULL_STACK=1` is clearer because it matches the actual goal: one venv
  that supports avatar prep, inference, and `load_test.py`
- full-stack avatar prep needs full `mmcv`; `mmcv-lite` is not sufficient
  because `musetalk.utils.preprocessing` requires `mmcv._ext`

This is still not recommended for general autoscaled inference workers.

Validation reminder for these nodes:

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
- Torch reports the `cu121` build
- `torch.version.cuda` reports `12.1`

## Suggested MVP Rollout

For the first autoscaling pass:

1. keep the repo on the persistent Vast workspace
2. use `scripts/vast_onstart.sh` as the instance boot command
3. start with `PROFILE=baseline`
4. verify with `bash scripts/vast_server_ctl.sh status`
5. hit `http://<instance-ip>:8000/health`

Once that is stable, switch selected instances to:

```bash
PROFILE=throughput_record bash scripts/vast_onstart.sh
```

## Notes

- these helpers do not replace the canonical setup / launch scripts
- they exist only to make boot-time automation safe and idempotent on Vast.ai
- if you later bake the venv and model weights into a custom image, keep
  `AUTO_SETUP=0` and use `scripts/vast_onstart.sh` only as the startup wrapper
