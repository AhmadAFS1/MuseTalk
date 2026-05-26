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
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PROFILE="${PROFILE:-baseline}"

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

Launch the current MuseTalk TRT-stagewise HLS server using the exact
configured venv interpreter. This avoids mixed-venv launches.

Profiles:
  baseline           Conservative stable TRT-stagewise HLS baseline (default)
  throughput_record  GPU-aware widened-batch throughput branch
  vram_max           Alias for GPU-aware throughput defaults

Options:
  --profile NAME     Launch profile: baseline, throughput_record, or vram_max
  --host HOST        Bind host (default: $HOST)
  --port PORT        Bind port (default: $PORT)
  --venv-path PATH   Python venv path (default: $VENV_PATH)
  --repo-root PATH   MuseTalk repo root (default: $REPO_ROOT)
  --help             Show this help text

Examples:
  $SCRIPT_NAME
  $SCRIPT_NAME --profile throughput_record
  HOST=127.0.0.1 PORT=8010 $SCRIPT_NAME --profile baseline
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires a value"
      PROFILE="$2"
      shift 2
      ;;
    --host)
      [[ $# -ge 2 ]] || die "--host requires a value"
      HOST="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port requires a value"
      PORT="$2"
      shift 2
      ;;
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
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

VENV_PYTHON="$VENV_PATH/bin/python"
[[ -x "$VENV_PYTHON" ]] || die "Venv python not found: $VENV_PYTHON"
[[ -f "$REPO_ROOT/api_server.py" ]] || die "api_server.py not found under: $REPO_ROOT"

unset PYTORCH_CUDA_ALLOC_CONF

# Avoid mixed shell/runtime thread overrides from previous experiments.
unset MUSETALK_CPU_TUNING
unset MUSETALK_CPU_THREADS
unset MUSETALK_CPU_INTEROP_THREADS
unset MUSETALK_CPU_CV2_THREADS
unset MUSETALK_CPU_NUMA_NODE
unset MUSETALK_CPU_AFFINITY

: "${MUSETALK_COMPILE:=0}"
: "${MUSETALK_COMPILE_UNET:=0}"
: "${MUSETALK_COMPILE_VAE:=0}"
: "${MUSETALK_WARM_RUNTIME:=1}"
: "${MUSETALK_TRT_ENABLED:=1}"
: "${MUSETALK_VAE_BACKEND:=trt_stagewise}"
: "${MUSETALK_TRT_FALLBACK:=0}"
: "${MUSETALK_TRT_STAGEWISE_TORCH_EXECUTED_OPS:=native_group_norm}"
: "${MUSETALK_TRT_STAGEWISE_TORCH_STAGES:=}"
: "${MUSETALK_TRT_STAGEWISE_PRECISION:=fp16}"
: "${MUSETALK_TRT_STAGEWISE_INT8_STAGES:=}"
: "${MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR:=}"
: "${MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_BATCHES:=8}"
: "${MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO:=minmax}"
: "${MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR:=./models/tensorrt/stagewise_int8_calibration_cache}"
: "${MUSETALK_TRT_STAGEWISE_INT8_USE_CACHE:=1}"
: "${MUSETALK_TRT_STAGEWISE_INT8_FRONTEND:=onnx_qdq}"
: "${MUSETALK_TRT_STAGEWISE_INT8_ENABLED_PRECISIONS:=int8}"
: "${MUSETALK_TRT_STAGEWISE_INT8_MIN_BLOCK_SIZE:=1}"
: "${MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_FULL_COMPILATION:=0}"
: "${MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_CALIBRATION_CACHE:=1}"
: "${MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_FORMAT:=tensor}"
: "${MUSETALK_TRT_STAGEWISE_INT8_TORCH_EXECUTED_OPS:=group_norm}"
: "${MUSETALK_TRT_STAGEWISE_INT8_ALLOW_UNSAFE_STAGES:=0}"
: "${MUSETALK_VAE_CALIBRATION_CAPTURE:=0}"
: "${MUSETALK_VAE_CALIBRATION_DIR:=./calibration/vae_decoder}"
: "${MUSETALK_VAE_CALIBRATION_MAX_BATCHES:=128}"

: "${WEBRTC_SYNC_MODE:=strict_fifo}"
: "${WEBRTC_VIDEO_PREBUFFER_SECONDS:=2.0}"
: "${WEBRTC_AUDIO_PREBUFFER_SECONDS:=0.0}"
: "${WEBRTC_ADAPTIVE_FPS:=0}"

: "${AVATAR_CACHE_MAX_AVATARS:=0}"
: "${AVATAR_CACHE_TTL_SECONDS:=3600}"

: "${HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS:=999}"
: "${HLS_STARTUP_CHUNK_DURATION_SECONDS:=0.5}"
: "${HLS_STARTUP_CHUNK_COUNT:=1}"
: "${HLS_PREP_WORKERS:=8}"
: "${HLS_COMPOSE_WORKERS:=8}"
: "${HLS_ENCODE_WORKERS:=8}"
: "${HLS_MAX_PENDING_JOBS:=24}"

: "${HLS_CHUNK_VIDEO_ENCODER:=libx264}"
: "${HLS_CHUNK_ENCODER_PRESET:=ultrafast}"
: "${HLS_CHUNK_ENCODER_CRF:=28}"
: "${HLS_PERSISTENT_SEGMENTER:=0}"
: "${HLS_CHUNK_PREPARE_AUDIO_SIDECAR:=1}"

: "${MUSETALK_WHISPER_SEGMENT_BATCH_SIZE:=4}"
: "${MUSETALK_AVATAR_LOAD_WORKERS:=8}"
: "${PYTHONFAULTHANDLER:=1}"
: "${PYTHONUNBUFFERED:=1}"

unset HLS_CHUNK_ENCODER_TUNE
unset HLS_CHUNK_ENCODER_QP

apply_gpu_aware_defaults() {
  local profile_name="$1"
  local assignments
  assignments="$(
    cd "$REPO_ROOT"
    PROFILE="$profile_name" "$VENV_PYTHON" - <<'PY'
import os
import shlex

from scripts.concurrent_gpu_manager import (
    default_reserved_memory_gb,
    detect_total_gpu_memory_gb,
    recommended_scheduler_batch_config,
)

profile = os.getenv("PROFILE", "baseline")
total_gb, source = detect_total_gpu_memory_gb(gpu_id=0)
reserved_gb = default_reserved_memory_gb(total_gb)
recommended = recommended_scheduler_batch_config(total_gb, profile=profile)
available_gb = max(1.0, total_gb - reserved_gb)
cache_mb = int(max(6000, min(24000, available_gb * 1024 * 0.75)))

defaults = {
    "PROFILE": profile,
    "GPU_TOTAL_MEMORY_GB": f"{total_gb:.1f}",
    "GPU_RESERVED_MEMORY_GB": f"{reserved_gb:.1f}",
    "HLS_SCHEDULER_MAX_BATCH": str(recommended["max_combined_batch_size"]),
    "HLS_SCHEDULER_FIXED_BATCH_SIZES": ",".join(str(v) for v in recommended["fixed_batch_sizes"]),
    "HLS_SCHEDULER_STARTUP_SLICE_SIZE": str(recommended["startup_slice_size"]),
    "MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES": ",".join(str(v) for v in recommended["warmup_batches"]),
    "AVATAR_CACHE_MAX_MEMORY_MB": str(cache_mb),
    "GPU_MEMORY_DETECTION_SOURCE": source,
}

for name, default in defaults.items():
    value = os.getenv(name) or default
    print(f"export {name}={shlex.quote(str(value))}")
PY
  )"
  eval "$assignments"
}

case "$PROFILE" in
  baseline|throughput_record|vram_max)
    apply_gpu_aware_defaults "$PROFILE"
    ;;
  *)
    die "Unsupported profile: $PROFILE"
    ;;
esac

export MUSETALK_COMPILE
export MUSETALK_COMPILE_UNET
export MUSETALK_COMPILE_VAE
export MUSETALK_WARM_RUNTIME
export MUSETALK_TRT_ENABLED
export MUSETALK_VAE_BACKEND
export MUSETALK_TRT_FALLBACK
export MUSETALK_TRT_STAGEWISE_TORCH_EXECUTED_OPS
export MUSETALK_TRT_STAGEWISE_TORCH_STAGES
export MUSETALK_TRT_STAGEWISE_PRECISION
export MUSETALK_TRT_STAGEWISE_INT8_STAGES
export MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR
export MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_BATCHES
export MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO
export MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR
export MUSETALK_TRT_STAGEWISE_INT8_USE_CACHE
export MUSETALK_TRT_STAGEWISE_INT8_FRONTEND
export MUSETALK_TRT_STAGEWISE_INT8_ENABLED_PRECISIONS
export MUSETALK_TRT_STAGEWISE_INT8_MIN_BLOCK_SIZE
export MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_FULL_COMPILATION
export MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_CALIBRATION_CACHE
export MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_FORMAT
export MUSETALK_TRT_STAGEWISE_INT8_TORCH_EXECUTED_OPS
export MUSETALK_TRT_STAGEWISE_INT8_ALLOW_UNSAFE_STAGES
export MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES
export MUSETALK_VAE_CALIBRATION_CAPTURE
export MUSETALK_VAE_CALIBRATION_DIR
export MUSETALK_VAE_CALIBRATION_MAX_BATCHES
export WEBRTC_SYNC_MODE
export WEBRTC_VIDEO_PREBUFFER_SECONDS
export WEBRTC_AUDIO_PREBUFFER_SECONDS
export WEBRTC_ADAPTIVE_FPS
export AVATAR_CACHE_MAX_AVATARS
export AVATAR_CACHE_MAX_MEMORY_MB
export AVATAR_CACHE_TTL_SECONDS
export HLS_SCHEDULER_MAX_BATCH
export HLS_SCHEDULER_FIXED_BATCH_SIZES
export HLS_SCHEDULER_STARTUP_SLICE_SIZE
export HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS
export HLS_STARTUP_CHUNK_DURATION_SECONDS
export HLS_STARTUP_CHUNK_COUNT
export HLS_PREP_WORKERS
export HLS_COMPOSE_WORKERS
export HLS_ENCODE_WORKERS
export HLS_MAX_PENDING_JOBS
export HLS_CHUNK_VIDEO_ENCODER
export HLS_CHUNK_ENCODER_PRESET
export HLS_CHUNK_ENCODER_CRF
export HLS_PERSISTENT_SEGMENTER
export HLS_CHUNK_PREPARE_AUDIO_SIDECAR
export MUSETALK_WHISPER_SEGMENT_BATCH_SIZE
export MUSETALK_AVATAR_LOAD_WORKERS
export PYTHONFAULTHANDLER
export PYTHONUNBUFFERED
export PROFILE
export GPU_TOTAL_MEMORY_GB
export GPU_RESERVED_MEMORY_GB
export GPU_MEMORY_DETECTION_SOURCE

log "Launching MuseTalk TRT-stagewise server"
log "profile=$PROFILE host=$HOST port=$PORT"
log "python=$VENV_PYTHON"
log "GPU_TOTAL_MEMORY_GB=$GPU_TOTAL_MEMORY_GB"
log "GPU_RESERVED_MEMORY_GB=$GPU_RESERVED_MEMORY_GB"
log "GPU_MEMORY_DETECTION_SOURCE=${GPU_MEMORY_DETECTION_SOURCE:-unknown}"
log "HLS_SCHEDULER_MAX_BATCH=$HLS_SCHEDULER_MAX_BATCH"
log "HLS_SCHEDULER_FIXED_BATCH_SIZES=$HLS_SCHEDULER_FIXED_BATCH_SIZES"
log "MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=$MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES"
log "MUSETALK_TRT_STAGEWISE_PRECISION=$MUSETALK_TRT_STAGEWISE_PRECISION"
log "MUSETALK_TRT_STAGEWISE_INT8_STAGES=${MUSETALK_TRT_STAGEWISE_INT8_STAGES:-default}"
log "MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR=${MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR:-${MUSETALK_VAE_CALIBRATION_DIR}}"
log "MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO=$MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO"
log "MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR=$MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR"
log "MUSETALK_VAE_CALIBRATION_CAPTURE=$MUSETALK_VAE_CALIBRATION_CAPTURE"
log "WEBRTC_H264_ENCODER=${WEBRTC_H264_ENCODER:-h264_nvenc(default)}"
log "WEBRTC_ICE_TRANSPORT_POLICY=${WEBRTC_ICE_TRANSPORT_POLICY:-all(default)}"
log "WEBRTC_STUN_URLS=${WEBRTC_STUN_URLS:-stun:stun.l.google.com:19302(default)}"
log "WEBRTC_TURN_URLS=${WEBRTC_TURN_URLS:-}"
log "WEBRTC_SERVER_TURN_URLS=${WEBRTC_SERVER_TURN_URLS:-}"
log "WEBRTC_SYNC_MODE=$WEBRTC_SYNC_MODE"
log "WEBRTC_VIDEO_PREBUFFER_SECONDS=$WEBRTC_VIDEO_PREBUFFER_SECONDS"
log "WEBRTC_ADAPTIVE_FPS=$WEBRTC_ADAPTIVE_FPS"
log "PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER PYTHONUNBUFFERED=$PYTHONUNBUFFERED"

cd "$REPO_ROOT"

child_pid=""
forward_signal() {
  local sig="$1"
  log "Received $sig; forwarding to api_server.py pid=${child_pid:-unknown}"
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" >/dev/null 2>&1; then
    kill "-$sig" "$child_pid" >/dev/null 2>&1 || true
  fi
}

"$VENV_PYTHON" api_server.py --host "$HOST" --port "$PORT" &
child_pid=$!
log "api_server.py started pid=$child_pid wrapper_pid=$$"

trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

set +e
wait "$child_pid"
rc=$?
set -e

trap - TERM INT

if (( rc >= 128 )); then
  signal_number=$((rc - 128))
  signal_name="$(kill -l "$signal_number" 2>/dev/null || true)"
  log "api_server.py exited with code=$rc signal=${signal_name:-$signal_number}"
else
  log "api_server.py exited with code=$rc"
fi

exit "$rc"
