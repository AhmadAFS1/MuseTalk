#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE:-/workspace}"
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/.venvs/musetalk_trt_stagewise}"
PROFILE="${PROFILE:-baseline}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
AUTO_SETUP="${AUTO_SETUP:-1}"
SETUP_CLEAN="${SETUP_CLEAN:-0}"
SETUP_SKIP_APT="${SETUP_SKIP_APT:-auto}"
SETUP_SKIP_WEIGHTS="${SETUP_SKIP_WEIGHTS:-0}"
SETUP_FULL_STACK="${SETUP_FULL_STACK:-0}"
SETUP_INSTALL_AVATAR_PREP_DEPS="${SETUP_INSTALL_AVATAR_PREP_DEPS:-0}"

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
}

die() {
  printf '[%s] ERROR: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

env_flag_is_true() {
  local value="${1:-}"
  case "${value,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

full_stack_requested() {
  env_flag_is_true "$SETUP_FULL_STACK"
}

avatar_prep_requested() {
  full_stack_requested || env_flag_is_true "$SETUP_INSTALL_AVATAR_PREP_DEPS"
}

server_runtime_imports_complete() {
  [[ -x "$VENV_PATH/bin/python" ]] || return 1

  (
    cd "$REPO_ROOT"
    "$VENV_PATH/bin/python" - <<'PY' >/dev/null 2>&1
import api_server
import aiofiles
import aiohttp
import av
import fastapi
import ffmpeg
import imageio
import librosa
import multipart
import omegaconf
import soundfile
import tensorrt
import torch
import torch_tensorrt
import uvicorn

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() returned False")
PY
  )
}

avatar_prep_runtime_imports_complete() {
  [[ -x "$VENV_PATH/bin/python" ]] || return 1

  (
    cd "$REPO_ROOT"
    "$VENV_PATH/bin/python" - <<'PY' >/dev/null 2>&1
import mmcv
import mmcv._ext
import mmdet
import mmengine
import mmpose
from musetalk.utils.preprocessing import get_landmark_and_bbox
PY
  )
}

setup_complete() {
  [[ -x "$VENV_PATH/bin/python" ]] || return 1

  local required=(
    "$REPO_ROOT/api_server.py"
    "$REPO_ROOT/models/musetalkV15/unet.pth"
    "$REPO_ROOT/models/sd-vae/diffusion_pytorch_model.bin"
    "$REPO_ROOT/models/whisper/pytorch_model.bin"
    "$REPO_ROOT/models/face-parse-bisent/79999_iter.pth"
  )

  local path
  for path in "${required[@]}"; do
    [[ -e "$path" ]] || return 1
  done

  server_runtime_imports_complete || return 1
}

avatar_prep_setup_complete() {
  setup_complete || return 1

  local required=(
    "$REPO_ROOT/models/dwpose/dw-ll_ucoco_384.pth"
    "$REPO_ROOT/models/syncnet/latentsync_syncnet.pt"
  )

  local path
  for path in "${required[@]}"; do
    [[ -e "$path" ]] || return 1
  done

  avatar_prep_runtime_imports_complete || return 1
}

run_setup_if_needed() {
  if ! env_flag_is_true "$AUTO_SETUP"; then
    log "AUTO_SETUP disabled"
    if avatar_prep_requested; then
      avatar_prep_setup_complete || die "AUTO_SETUP=0 but avatar-prep runtime validation failed"
    else
      setup_complete || die "AUTO_SETUP=0 but required runtime files are missing"
    fi
    return 0
  fi

  if avatar_prep_requested; then
    if avatar_prep_setup_complete; then
      log "Existing full-stack setup looks valid; skipping bootstrap"
      return 0
    fi
    if setup_complete; then
      log "Server runtime is present, but full-stack avatar-prep support is incomplete"
    fi
  else
    if setup_complete; then
      log "Existing setup looks valid; skipping bootstrap"
      return 0
    fi
  fi

  local setup_args=("--venv-path" "$VENV_PATH")
  local rebuild_with_clean=0

  if [[ -e "$VENV_PATH" ]]; then
    log "Bootstrap requested and an existing venv is present; rebuilding with --clean"
    rebuild_with_clean=1
  fi

  if env_flag_is_true "$SETUP_CLEAN"; then
    rebuild_with_clean=1
  fi

  if (( rebuild_with_clean )); then
    setup_args+=("--clean")
  fi

  case "${SETUP_SKIP_APT,,}" in
    auto)
      if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
        setup_args+=("--skip-apt")
      fi
      ;;
    1|true|yes|on)
      setup_args+=("--skip-apt")
      ;;
    0|false|no|off)
      ;;
    *)
      die "Unsupported SETUP_SKIP_APT value: $SETUP_SKIP_APT"
      ;;
  esac

  if env_flag_is_true "$SETUP_SKIP_WEIGHTS"; then
    setup_args+=("--skip-weights")
  fi

  if full_stack_requested; then
    setup_args+=("--full-stack")
  elif env_flag_is_true "$SETUP_INSTALL_AVATAR_PREP_DEPS"; then
    setup_args+=("--install-avatar-prep-deps")
  fi

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    setup_args+=("--python-bin" "$PYTHON_BIN")
  fi

  if [[ -n "${ARTIFACT_DIR:-}" ]]; then
    setup_args+=("--artifact-dir" "$ARTIFACT_DIR")
  fi

  log "Running setup_musetalk.sh ${setup_args[*]}"
  bash "$REPO_ROOT/setup_musetalk.sh" "${setup_args[@]}"
}

main() {
  local setup_mode="server-only"
  if full_stack_requested; then
    setup_mode="full-stack"
  elif avatar_prep_requested; then
    setup_mode="server+avatar-prep"
  fi

  log "Vast.ai MuseTalk on-start begin"
  log "repo=$REPO_ROOT"
  log "workspace=$WORKSPACE_ROOT"
  log "venv=$VENV_PATH"
  log "profile=$PROFILE host=$HOST port=$PORT"
  log "setup_mode=$setup_mode"

  run_setup_if_needed

  PROFILE="$PROFILE" \
  HOST="$HOST" \
  PORT="$PORT" \
  REPO_ROOT="$REPO_ROOT" \
  VENV_PATH="$VENV_PATH" \
  bash "$REPO_ROOT/scripts/vast_server_ctl.sh" start

  log "Vast.ai MuseTalk on-start complete"
}

main "$@"
