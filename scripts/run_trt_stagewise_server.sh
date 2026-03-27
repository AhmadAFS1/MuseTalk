#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_PATH="${VENV_PATH:-/content/py310_trt_exp}"
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
/content/py310_trt_exp interpreter. This avoids mixed-venv launches.

Profiles:
  baseline           Conservative stable TRT-stagewise HLS baseline (default)
  throughput_record  Current widened-batch throughput branch

Options:
  --profile NAME     Launch profile: baseline or throughput_record
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

: "${AVATAR_CACHE_MAX_AVATARS:=0}"
: "${AVATAR_CACHE_MAX_MEMORY_MB:=12000}"
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

unset HLS_CHUNK_ENCODER_TUNE
unset HLS_CHUNK_ENCODER_QP

case "$PROFILE" in
  baseline)
    : "${HLS_SCHEDULER_MAX_BATCH:=4}"
    : "${HLS_SCHEDULER_FIXED_BATCH_SIZES:=4}"
    : "${HLS_SCHEDULER_STARTUP_SLICE_SIZE:=4}"
    : "${MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES:=4}"
    ;;
  throughput_record)
    # Current best average-throughput branch on the RTX 3090:
    # max_batch=16 with request batch_size=8 in load_test.py.
    : "${HLS_SCHEDULER_MAX_BATCH:=16}"
    : "${HLS_SCHEDULER_FIXED_BATCH_SIZES:=4,8,16}"
    : "${HLS_SCHEDULER_STARTUP_SLICE_SIZE:=4}"
    # Warm 8 and 16 by default to match the widened-batch branch without
    # forcing a 4+8+16 warmup that previously OOM'd on 24 GB cards.
    : "${MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES:=8,16}"
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
export MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES
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

log "Launching MuseTalk TRT-stagewise server"
log "profile=$PROFILE host=$HOST port=$PORT"
log "python=$VENV_PYTHON"
log "HLS_SCHEDULER_MAX_BATCH=$HLS_SCHEDULER_MAX_BATCH"
log "HLS_SCHEDULER_FIXED_BATCH_SIZES=$HLS_SCHEDULER_FIXED_BATCH_SIZES"
log "MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=$MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES"

cd "$REPO_ROOT"
exec "$VENV_PYTHON" api_server.py --host "$HOST" --port "$PORT"
