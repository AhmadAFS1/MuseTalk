#!/usr/bin/env bash
set -euo pipefail

# Added helper script:
# Creates the separate TensorRT experiment venv without touching the stable
# /content/py310 environment unless the caller explicitly asks to clean or
# remove caches.

SCRIPT_NAME="$(basename "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_PATH="${VENV_PATH:-/content/py310_trt_exp}"
REPO_ROOT="${REPO_ROOT:-/content/MuseTalk}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/content/MuseTalk/models/tensorrt_altenv}"
RUNTIME_REQUIREMENTS="${RUNTIME_REQUIREMENTS:-}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"

TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TORCH_TENSORRT_VERSION="${TORCH_TENSORRT_VERSION:-2.5.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
NVIDIA_EXTRA_INDEX_URL="${NVIDIA_EXTRA_INDEX_URL:-https://pypi.nvidia.com}"

CLEAN=0
CLEAR_PIP_CACHE=0
INSTALL_RUNTIME_DEPS=0

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

Creates the separate TensorRT experiment environment described in
current_tensorrt_environment_plan.md.

Options:
  --venv-path PATH           Target venv path (default: $VENV_PATH)
  --repo-root PATH           MuseTalk repo root (default: $REPO_ROOT)
  --artifact-dir PATH        TensorRT artifact output dir (default: $ARTIFACT_DIR)
  --python-bin PATH          Python 3.10 interpreter to use (default: $PYTHON_BIN)
  --runtime-requirements P   Requirements file for optional runtime deps
  --install-runtime-deps     Install repo runtime deps after backend validation
  --clean                    Remove an existing venv at --venv-path first
  --clear-pip-cache          Remove the pip cache dir before installing
  --help                     Show this help text

Environment overrides:
  TORCH_VERSION              Default: $TORCH_VERSION
  TORCHVISION_VERSION        Default: $TORCHVISION_VERSION
  TORCHAUDIO_VERSION         Default: $TORCHAUDIO_VERSION
  TORCH_TENSORRT_VERSION     Default: $TORCH_TENSORRT_VERSION
  PYTORCH_INDEX_URL          Default: $PYTORCH_INDEX_URL
  NVIDIA_EXTRA_INDEX_URL     Default: $NVIDIA_EXTRA_INDEX_URL

Examples:
  $SCRIPT_NAME --clean --clear-pip-cache
  $SCRIPT_NAME --clean --install-runtime-deps
  VENV_PATH=/content/py310_trt_exp_alt $SCRIPT_NAME --clean
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
    --repo-root)
      [[ $# -ge 2 ]] || die "--repo-root requires a value"
      REPO_ROOT="$2"
      shift 2
      ;;
    --artifact-dir)
      [[ $# -ge 2 ]] || die "--artifact-dir requires a value"
      ARTIFACT_DIR="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || die "--python-bin requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --runtime-requirements)
      [[ $# -ge 2 ]] || die "--runtime-requirements requires a value"
      RUNTIME_REQUIREMENTS="$2"
      shift 2
      ;;
    --install-runtime-deps)
      INSTALL_RUNTIME_DEPS=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --clear-pip-cache)
      CLEAR_PIP_CACHE=1
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

require_command "$PYTHON_BIN"
require_command rm
require_command mkdir

if [[ -z "$RUNTIME_REQUIREMENTS" ]]; then
  RUNTIME_REQUIREMENTS="$REPO_ROOT/requirements.txt"
fi

if ! "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY
then
  die "Expected a Python 3.10 interpreter at: $PYTHON_BIN"
fi

if [[ ! -d "$REPO_ROOT" ]]; then
  die "Repo root not found: $REPO_ROOT"
fi

if [[ $CLEAR_PIP_CACHE -eq 1 ]]; then
  if [[ -d "$PIP_CACHE_DIR" ]]; then
    log "Removing pip cache: $PIP_CACHE_DIR"
    rm -rf "$PIP_CACHE_DIR"
  else
    log "Pip cache dir not present, skipping: $PIP_CACHE_DIR"
  fi
fi

if [[ -e "$VENV_PATH" ]]; then
  if [[ $CLEAN -eq 1 ]]; then
    log "Removing existing venv: $VENV_PATH"
    rm -rf "$VENV_PATH"
  else
    die "Venv already exists at $VENV_PATH. Re-run with --clean to recreate it."
  fi
fi

log "Creating TensorRT artifact directory: $ARTIFACT_DIR"
mkdir -p "$ARTIFACT_DIR"

log "Creating venv: $VENV_PATH"
"$PYTHON_BIN" -m venv "$VENV_PATH"

VENV_PYTHON="$VENV_PATH/bin/python"

[[ -x "$VENV_PYTHON" ]] || die "Venv python not found after creation: $VENV_PYTHON"

log "Upgrading pip, setuptools, and wheel"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

log "Installing PyTorch CUDA 12.1 wheel set"
"$VENV_PYTHON" -m pip install --no-cache-dir \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "$PYTORCH_INDEX_URL"

log "Installing torch-tensorrt pinned stack"
"$VENV_PYTHON" -m pip install --no-cache-dir \
  --extra-index-url "$NVIDIA_EXTRA_INDEX_URL" \
  "torch-tensorrt==$TORCH_TENSORRT_VERSION"

log "Validating backend imports and CUDA registration"
"$VENV_PYTHON" - <<'PY'
import torch
import torch_tensorrt
import tensorrt

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() returned False")

engine_class = getattr(torch.classes.tensorrt, "Engine")
device_name = torch.cuda.get_device_name(0)

print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("torch_tensorrt", torch_tensorrt.__version__)
print("tensorrt", tensorrt.__version__)
print("cuda_device", device_name)
print("engine_class", engine_class)
PY

if [[ $INSTALL_RUNTIME_DEPS -eq 1 ]]; then
  [[ -f "$RUNTIME_REQUIREMENTS" ]] || die "Runtime requirements file not found: $RUNTIME_REQUIREMENTS"
  log "Installing MuseTalk runtime dependencies from: $RUNTIME_REQUIREMENTS"
  "$VENV_PYTHON" -m pip install --no-cache-dir -r "$RUNTIME_REQUIREMENTS"
else
  log "Skipping runtime dependency install. Use --install-runtime-deps to include repo requirements."
fi

log "Setup complete"
printf '\n'
printf 'Next steps:\n'
printf '  source %s/bin/activate\n' "$VENV_PATH"
printf '  python scripts/tensorrt_export.py ...\n'
printf '  python scripts/benchmark_pipeline.py ...\n'
printf '\n'
printf 'TensorRT artifacts should be kept under: %s\n' "$ARTIFACT_DIR"
