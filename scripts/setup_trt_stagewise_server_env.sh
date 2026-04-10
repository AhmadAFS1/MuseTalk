#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE:-}"
if [[ -z "$WORKSPACE_ROOT" ]]; then
  if [[ "$REPO_ROOT" == /workspace/* || "$REPO_ROOT" == "/workspace" ]]; then
    WORKSPACE_ROOT="/workspace"
  elif [[ "$REPO_ROOT" == /content/* || "$REPO_ROOT" == "/content" ]]; then
    WORKSPACE_ROOT="/content"
  else
    WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
  fi
fi
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/.venvs/musetalk_trt_stagewise}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$REPO_ROOT/models/tensorrt_altenv_bs32}"

CLEAN=0
SKIP_APT=0
SKIP_WEIGHTS=0
INSTALL_AVATAR_PREP_DEPS=0
FULL_STACK=0

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
}

die() {
  printf '[%s] ERROR: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options]

Create the current MuseTalk TRT-stagewise HLS server environment from scratch.

This wrapper installs the system packages needed by the current server shape,
builds the configured TRT-stagewise venv through setup_trt_experiment_env.sh,
downloads/validates model weights, and runs a final import smoke test.

Options:
  --venv-path PATH      Target venv path (default: $VENV_PATH)
  --python-bin PATH     Python 3.10 interpreter to use (default: $PYTHON_BIN)
  --artifact-dir PATH   TensorRT artifact dir (default: $ARTIFACT_DIR)
  --clean               Recreate the venv from scratch
  --skip-apt            Skip apt-get system package installation
  --skip-weights        Skip download_weights.sh and only validate required files
  --full-stack          Build one venv with both server and avatar-prep deps
  --install-avatar-prep-deps
                        Install optional mmpose/mmcv deps for avatar prep
  --help                Show this help text

Examples:
  $SCRIPT_NAME --clean
  $SCRIPT_NAME --clean --full-stack
  $SCRIPT_NAME --clean --skip-apt
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-path)
      [[ $# -ge 2 ]] || die "--venv-path requires a value"
      VENV_PATH="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || die "--python-bin requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --artifact-dir)
      [[ $# -ge 2 ]] || die "--artifact-dir requires a value"
      ARTIFACT_DIR="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --skip-weights)
      SKIP_WEIGHTS=1
      shift
      ;;
    --full-stack)
      FULL_STACK=1
      INSTALL_AVATAR_PREP_DEPS=1
      shift
      ;;
    --install-avatar-prep-deps)
      INSTALL_AVATAR_PREP_DEPS=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

require_command bash

if [[ $SKIP_APT -eq 0 ]]; then
  require_command apt-get
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    die "apt install requested, but this shell is not running as root. Re-run as root or use --skip-apt."
  fi

  log "Installing required system packages"
  apt-get update -y
  apt-get install -y \
    python3.10 \
    python3.10-venv \
    ffmpeg \
    git \
    curl \
    build-essential \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1
fi

require_command "$PYTHON_BIN"
require_command ffmpeg
require_command curl
require_command git

log "Creating the pinned TRT experiment venv"
SETUP_ARGS=(
  --venv-path "$VENV_PATH"
  --repo-root "$REPO_ROOT"
  --python-bin "$PYTHON_BIN"
  --artifact-dir "$ARTIFACT_DIR"
  --install-server-deps
)
if [[ $CLEAN -eq 1 ]]; then
  SETUP_ARGS+=(--clean)
fi
if [[ $INSTALL_AVATAR_PREP_DEPS -eq 1 ]]; then
  if [[ $FULL_STACK -eq 1 ]]; then
    log "Full-stack mode enabled: this venv will install both server and avatar-prep deps"
    SETUP_ARGS+=(--full-stack)
  else
    SETUP_ARGS+=(--install-avatar-prep-deps)
  fi
fi
bash "$SCRIPT_DIR/setup_trt_experiment_env.sh" "${SETUP_ARGS[@]}"

VENV_PYTHON="$VENV_PATH/bin/python"
[[ -x "$VENV_PYTHON" ]] || die "Expected venv python at: $VENV_PYTHON"

if [[ $SKIP_WEIGHTS -eq 0 ]]; then
  log "Downloading / validating model weights"
  (
    cd "$REPO_ROOT"
    export PATH="$VENV_PATH/bin:$PATH"
    bash ./download_weights.sh
  )
else
  log "Skipping weight download by request"
fi

log "Validating required model files"
"$VENV_PYTHON" - <<PY
from pathlib import Path

repo = Path(${REPO_ROOT@Q})
required = [
    repo / "models/musetalk/musetalk.json",
    repo / "models/musetalk/pytorch_model.bin",
    repo / "models/musetalkV15/musetalk.json",
    repo / "models/musetalkV15/unet.pth",
    repo / "models/sd-vae/config.json",
    repo / "models/sd-vae/diffusion_pytorch_model.bin",
    repo / "models/whisper/config.json",
    repo / "models/whisper/pytorch_model.bin",
    repo / "models/whisper/preprocessor_config.json",
    repo / "models/dwpose/dw-ll_ucoco_384.pth",
    repo / "models/syncnet/latentsync_syncnet.pt",
    repo / "models/face_detection/s3fd.pth",
    repo / "models/face-parse-bisent/79999_iter.pth",
    repo / "models/face-parse-bisent/resnet18-5c106cde.pth",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing required model files:\n" + "\n".join(missing))
print("Validated", len(required), "required model files")
PY

if [[ $INSTALL_AVATAR_PREP_DEPS -eq 1 ]]; then
  log "Running avatar-prep import smoke test"
  (
    cd "$REPO_ROOT"
    "$VENV_PYTHON" - <<'PY'
import mmcv
import mmcv._ext
import mmdet
import mmengine
import mmpose
from musetalk.utils.preprocessing import get_landmark_and_bbox

print("mmengine", mmengine.__version__)
print("mmcv", mmcv.__version__)
print("mmcv._ext", "available")
print("mmdet", mmdet.__version__)
print("mmpose", mmpose.__version__)
print("get_landmark_and_bbox", callable(get_landmark_and_bbox))
print("avatar prep import OK")
PY
  )
fi

log "Running TRT-stagewise server import smoke test"
(
  cd "$REPO_ROOT"
  "$VENV_PYTHON" - <<'PY'
import os

import api_server
import torch
import torch_tensorrt
import tensorrt

from scripts.trt_runtime import _stagewise_warmup_batches

print("api_server import OK")
print("torch", torch.__version__)
print("torch_tensorrt", torch_tensorrt.__version__)
print("tensorrt", tensorrt.__version__)
print("cuda_available", torch.cuda.is_available())
print("default_stagewise_warmup_batches", _stagewise_warmup_batches())
PY
)

printf '\n'
printf 'Fresh-server setup complete.\n'
printf 'Recommended next step:\n'
printf '  bash %s/scripts/run_trt_stagewise_server.sh --profile baseline\n' "$REPO_ROOT"
printf '\n'
printf 'Optional throughput profile:\n'
printf '  bash %s/scripts/run_trt_stagewise_server.sh --profile throughput_record\n' "$REPO_ROOT"
