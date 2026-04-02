#!/usr/bin/env bash
set -euo pipefail

# Added helper script:
# Creates the separate TensorRT experiment venv without touching any legacy
# stable environment unless the caller explicitly asks to clean or remove
# caches.

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
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
ARTIFACT_DIR="${ARTIFACT_DIR:-$REPO_ROOT/models/tensorrt_altenv_bs32}"
RUNTIME_REQUIREMENTS="${RUNTIME_REQUIREMENTS:-}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"

TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TORCH_TENSORRT_VERSION="${TORCH_TENSORRT_VERSION:-2.5.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
NVIDIA_EXTRA_INDEX_URL="${NVIDIA_EXTRA_INDEX_URL:-https://pypi.nvidia.com}"

NUMPY_VERSION="${NUMPY_VERSION:-1.23.5}"
OPENCV_VERSION="${OPENCV_VERSION:-4.9.0.80}"
DIFFUSERS_VERSION="${DIFFUSERS_VERSION:-0.30.2}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.39.2}"
TOKENIZERS_VERSION="${TOKENIZERS_VERSION:-0.15.2}"
ACCELERATE_VERSION="${ACCELERATE_VERSION:-0.28.0}"
HUGGINGFACE_HUB_VERSION="${HUGGINGFACE_HUB_VERSION:-0.30.2}"
EINOPS_VERSION="${EINOPS_VERSION:-0.8.1}"
SAFETENSORS_VERSION="${SAFETENSORS_VERSION:-0.7.0}"
PILLOW_VERSION="${PILLOW_VERSION:-11.3.0}"

FASTAPI_VERSION="${FASTAPI_VERSION:-0.135.1}"
UVICORN_VERSION="${UVICORN_VERSION:-0.42.0}"
AIOHTTP_VERSION="${AIOHTTP_VERSION:-3.13.3}"
SOUNDFILE_VERSION="${SOUNDFILE_VERSION:-0.12.1}"
LIBROSA_VERSION="${LIBROSA_VERSION:-0.11.0}"
IMAGEIO_VERSION="${IMAGEIO_VERSION:-2.37.3}"
OMEGACONF_VERSION="${OMEGACONF_VERSION:-2.3.0}"
FFMPEG_PYTHON_VERSION="${FFMPEG_PYTHON_VERSION:-0.2.0}"
AIOFILES_VERSION="${AIOFILES_VERSION:-24.1.0}"
AV_VERSION="${AV_VERSION:-16.1.0}"
PYTHON_MULTIPART_VERSION="${PYTHON_MULTIPART_VERSION:-0.0.22}"
AIORTC_VERSION="${AIORTC_VERSION:-1.14.0}"
AIOICE_VERSION="${AIOICE_VERSION:-0.10.1}"
MMENGINE_VERSION="${MMENGINE_VERSION:-0.10.4}"
MMCV_VERSION="${MMCV_VERSION:-2.1.0}"
MMCV_LITE_VERSION="${MMCV_LITE_VERSION:-$MMCV_VERSION}"
MMDET_VERSION="${MMDET_VERSION:-3.2.0}"
MMPOSE_VERSION="${MMPOSE_VERSION:-1.3.1}"
OPENMIM_VERSION="${OPENMIM_VERSION:-0.3.9}"
CHUMPY_VERSION="${CHUMPY_VERSION:-0.70}"

CLEAN=0
CLEAR_PIP_CACHE=0
INSTALL_RUNTIME_DEPS=0
INSTALL_SERVER_DEPS=0
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

Creates the separate TensorRT experiment environment described in
current_tensorrt_environment_plan.md.

Options:
  --venv-path PATH           Target venv path (default: $VENV_PATH)
  --repo-root PATH           MuseTalk repo root (default: $REPO_ROOT)
  --artifact-dir PATH        TensorRT artifact output dir (default: $ARTIFACT_DIR)
  --python-bin PATH          Python 3.10 interpreter to use (default: $PYTHON_BIN)
  --runtime-requirements P   Requirements file for optional runtime deps
  --full-stack               Install both server deps and avatar-prep deps in
                             one venv
  --install-server-deps      Install the pinned HLS/api_server dependency set
  --install-avatar-prep-deps Install optional mmpose/mmcv deps for avatar prep
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
  NUMPY_VERSION              Default: $NUMPY_VERSION
  OPENCV_VERSION             Default: $OPENCV_VERSION
  HUGGINGFACE_HUB_VERSION    Default: $HUGGINGFACE_HUB_VERSION
  PYTHON_MULTIPART_VERSION   Default: $PYTHON_MULTIPART_VERSION
  AIORTC_VERSION             Default: $AIORTC_VERSION
  AIOICE_VERSION             Default: $AIOICE_VERSION
  MMENGINE_VERSION           Default: $MMENGINE_VERSION
  MMCV_VERSION               Default: $MMCV_VERSION
  MMCV_LITE_VERSION          Default: $MMCV_LITE_VERSION
  MMDET_VERSION              Default: $MMDET_VERSION
  MMPOSE_VERSION             Default: $MMPOSE_VERSION
  OPENMIM_VERSION            Default: $OPENMIM_VERSION
  CHUMPY_VERSION             Default: $CHUMPY_VERSION

Examples:
  $SCRIPT_NAME --clean --clear-pip-cache
  $SCRIPT_NAME --clean --install-server-deps
  VENV_PATH=/somewhere/.venvs/musetalk_trt_stagewise_alt $SCRIPT_NAME --clean
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

find_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    return 0
  fi
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    printf '%s\n' "${CUDA_HOME}/bin/nvcc"
    return 0
  fi
  return 1
}

cuda_toolkit_version() {
  local nvcc_bin
  nvcc_bin="$(find_nvcc)" || return 1
  "$nvcc_bin" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1
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
    --full-stack)
      FULL_STACK=1
      INSTALL_SERVER_DEPS=1
      INSTALL_AVATAR_PREP_DEPS=1
      shift
      ;;
    --install-runtime-deps)
      INSTALL_RUNTIME_DEPS=1
      shift
      ;;
    --install-server-deps)
      INSTALL_SERVER_DEPS=1
      shift
      ;;
    --install-avatar-prep-deps)
      INSTALL_AVATAR_PREP_DEPS=1
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

log "Installing pinned export + benchmark dependencies"
"$VENV_PYTHON" -m pip install --no-cache-dir \
  "numpy==$NUMPY_VERSION" \
  "opencv-python==$OPENCV_VERSION" \
  "diffusers==$DIFFUSERS_VERSION" \
  "transformers==$TRANSFORMERS_VERSION" \
  "tokenizers==$TOKENIZERS_VERSION" \
  "accelerate==$ACCELERATE_VERSION" \
  "huggingface_hub==$HUGGINGFACE_HUB_VERSION" \
  "einops==$EINOPS_VERSION" \
  "safetensors==$SAFETENSORS_VERSION" \
  "pillow==$PILLOW_VERSION"

log "Validating backend imports and CUDA registration"
"$VENV_PYTHON" - <<'PY'
import torch
import torch_tensorrt
import tensorrt
import cv2
import diffusers
import transformers
import huggingface_hub

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() returned False")

engine_class = getattr(torch.classes.tensorrt, "Engine")
device_name = torch.cuda.get_device_name(0)

print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("torch_tensorrt", torch_tensorrt.__version__)
print("tensorrt", tensorrt.__version__)
print("cv2", cv2.__version__)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("cuda_device", device_name)
print("engine_class", engine_class)
PY

if [[ $INSTALL_SERVER_DEPS -eq 1 ]]; then
  log "Installing pinned HLS/api_server dependency set"
  "$VENV_PYTHON" -m pip install --no-cache-dir \
    "fastapi==$FASTAPI_VERSION" \
    "uvicorn==$UVICORN_VERSION" \
    "aiohttp==$AIOHTTP_VERSION" \
    "soundfile==$SOUNDFILE_VERSION" \
    "librosa==$LIBROSA_VERSION" \
    "imageio[ffmpeg]==$IMAGEIO_VERSION" \
    "omegaconf==$OMEGACONF_VERSION" \
    "ffmpeg-python==$FFMPEG_PYTHON_VERSION" \
    "aiofiles==$AIOFILES_VERSION" \
    "av==$AV_VERSION" \
    "python-multipart==$PYTHON_MULTIPART_VERSION" \
    "aiortc==$AIORTC_VERSION" \
    "aioice==$AIOICE_VERSION"

  log "Validating HLS/api_server import path"
  "$VENV_PYTHON" - <<'PY'
import numpy
import cv2
import fastapi
import uvicorn
import aiohttp
import soundfile
import librosa
import imageio
import omegaconf
import ffmpeg
import aiofiles
import av
import multipart
import aiortc
import aioice

print("numpy", numpy.__version__)
print("cv2", cv2.__version__)
print("fastapi", fastapi.__version__)
print("uvicorn", uvicorn.__version__)
print("aiohttp", aiohttp.__version__)
print("soundfile", soundfile.__version__)
print("librosa", librosa.__version__)
print("imageio", imageio.__version__)
print("omegaconf", omegaconf.__version__)
print("av", av.__version__)
print("multipart", multipart.__version__)
print("aiortc", aiortc.__version__)
print("aioice", aioice.__version__)
print("all server dependency imports OK")
PY

  log "Validating api_server full import (includes transitive deps)"
  (
    cd "$REPO_ROOT"
    "$VENV_PYTHON" -c "import api_server; print('api_server import OK')"
  )
fi

if [[ $INSTALL_AVATAR_PREP_DEPS -eq 1 ]]; then
  TOOLKIT_CUDA_VERSION="$(cuda_toolkit_version || true)"
  TORCH_CUDA_VERSION="$("$VENV_PYTHON" - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)"

  if [[ -z "$TOOLKIT_CUDA_VERSION" ]]; then
    die "Avatar-prep deps require full mmcv, but no CUDA toolkit (nvcc) was found. Install a CUDA toolkit matching torch CUDA $TORCH_CUDA_VERSION or use an image that already includes it."
  fi

  if [[ "$TOOLKIT_CUDA_VERSION" != "$TORCH_CUDA_VERSION" ]]; then
    die "Avatar-prep deps require full mmcv, but local CUDA toolkit $TOOLKIT_CUDA_VERSION does not match torch CUDA $TORCH_CUDA_VERSION. Use a CUDA $TORCH_CUDA_VERSION image/toolkit for a single full-stack venv."
  fi

  log "Installing optional avatar-preparation deps (openmim + mmlab stack)"
  "$VENV_PYTHON" -m pip install --no-cache-dir \
    "openmim==$OPENMIM_VERSION" \
    "setuptools<81" \
    "ninja" \
    "psutil"

  "$VENV_PYTHON" -m mim install "mmengine==$MMENGINE_VERSION"
  if ! "$VENV_PYTHON" -m mim install "mmcv==$MMCV_VERSION"; then
    log "mim install mmcv failed; trying source build without build isolation"
    if ! "$VENV_PYTHON" -m pip install --no-cache-dir --no-build-isolation \
      "mmcv==$MMCV_VERSION"; then
      log "mmcv source build failed; falling back to mmcv-lite==$MMCV_LITE_VERSION"
      "$VENV_PYTHON" -m pip install --no-cache-dir \
        "mmcv-lite==$MMCV_LITE_VERSION"
    fi
  fi
  "$VENV_PYTHON" -m mim install "mmdet==$MMDET_VERSION"
  "$VENV_PYTHON" -m pip install --no-cache-dir \
    "chumpy==$CHUMPY_VERSION" \
    --no-build-isolation

  if ! "$VENV_PYTHON" -m mim install "mmpose==$MMPOSE_VERSION"; then
    log "mim install mmpose failed; falling back to pip"
    "$VENV_PYTHON" -m pip install --no-cache-dir \
      "mmpose==$MMPOSE_VERSION"
  fi

  log "Validating avatar-preparation imports"
  (
    cd "$REPO_ROOT"
    "$VENV_PYTHON" - <<'PY'
import mmengine
import mmcv
import mmdet
import mmpose
from musetalk.utils.preprocessing import get_landmark_and_bbox

print("mmengine", mmengine.__version__)
print("mmcv", mmcv.__version__)
print("mmdet", mmdet.__version__)
print("mmpose", mmpose.__version__)
print("avatar prep imports OK")
PY
  )
fi

if [[ $INSTALL_RUNTIME_DEPS -eq 1 ]]; then
  [[ -f "$RUNTIME_REQUIREMENTS" ]] || die "Runtime requirements file not found: $RUNTIME_REQUIREMENTS"
  log "Installing MuseTalk runtime dependencies from: $RUNTIME_REQUIREMENTS"
  log "Warning: full requirements are broader than the validated TRT HLS stack"
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
if [[ $INSTALL_SERVER_DEPS -eq 1 ]]; then
  printf '  python api_server.py --host 0.0.0.0 --port 8000\n'
  printf '  # Current validated HLS runtime shape: eager UNet + TRT VAE (leave MUSETALK_COMPILE=0)\n'
fi
printf '\n'
printf 'TensorRT artifacts should be kept under: %s\n' "$ARTIFACT_DIR"
