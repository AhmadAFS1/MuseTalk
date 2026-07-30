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
VALIDATE_ONLY=0

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

Default precision:
  VAE TRT-stagewise launches with the validated five-stage INT8 ONNX/QDQ profile
  when loaded from the generated TRT profile env. New Vast servers restore the
  artifact bundle first, then select the static batch-8 FP16 UNet TensorRT
  runtime with the batch-8 VAE INT8 cache.
  Set MUSETALK_TRT_STAGEWISE_PRECISION=fp16 and MUSETALK_TRT_UNET_ENABLED=0
  to opt out.

Options:
  --profile NAME     Launch profile: baseline, throughput_record, or vram_max
  --host HOST        Bind host (default: $HOST)
  --port PORT        Bind port (default: $PORT)
  --venv-path PATH   Python venv path (default: $VENV_PATH)
  --repo-root PATH   MuseTalk repo root (default: $REPO_ROOT)
  --validate-only    Validate dependencies/artifacts without starting the API
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
    --validate-only)
      VALIDATE_ONLY=1
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

MUSETALK_TRT_PROFILE_ENV_FILE="${MUSETALK_TRT_PROFILE_ENV_FILE:-$REPO_ROOT/.runtime/musetalk_trt_best.env}"
if [[ "${MUSETALK_TRT_PROFILE_ENV_LOAD:-1}" != "0" && -f "$MUSETALK_TRT_PROFILE_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$MUSETALK_TRT_PROFILE_ENV_FILE"
  set +a
  log "Loaded TRT runtime profile env: $MUSETALK_TRT_PROFILE_ENV_FILE"
fi

: "${MUSETALK_TRT_PROFILE_NAME:=}"
: "${MUSETALK_COMPILE:=0}"
: "${MUSETALK_COMPILE_UNET:=0}"
: "${MUSETALK_COMPILE_VAE:=0}"
: "${MUSETALK_WARM_RUNTIME:=1}"
: "${MUSETALK_TRT_ENABLED:=1}"
: "${MUSETALK_VAE_BACKEND:=trt_stagewise}"
: "${MUSETALK_TRT_FALLBACK:=0}"
: "${MUSETALK_UNET_BACKEND:=eager}"
: "${MUSETALK_TRT_UNET_ENABLED:=0}"
: "${MUSETALK_TRT_UNET_BUILD:=0}"
: "${MUSETALK_TRT_UNET_OUTPUT_DIR:=./models/tensorrt_unet_static_bs8_20260529}"
: "${MUSETALK_TRT_UNET_PATHS:=8:${MUSETALK_TRT_UNET_OUTPUT_DIR}/unet_trt.ts}"
: "${MUSETALK_TRT_UNET_PATH:=}"
: "${MUSETALK_TRT_UNET_META_PATH:=}"
: "${MUSETALK_TRT_UNET_ALLOW_UNVALIDATED:=0}"
: "${MUSETALK_TRT_UNET_CAPTURE_DIR:=./calibration/unet_static_8_16_20260529_1545}"
: "${MUSETALK_TRT_UNET_BUILD_BATCH_SIZES:=8}"
: "${MUSETALK_TRT_UNET_WORKSPACE_GB:=2}"
: "${MUSETALK_TRT_UNET_MIN_BLOCK_SIZE:=1}"
: "${MUSETALK_TRT_UNET_VALIDATE_LIMIT:=16}"
: "${MUSETALK_TRT_UNET_VALIDATE_PADDED_BATCH_SIZE:=8}"
: "${MUSETALK_TRT_UNET_VALIDATE_REPORT_PATH:=./tmp/unet_trt_static_bs8_validation_startup.json}"
: "${MUSETALK_TRT_STAGEWISE_TORCH_EXECUTED_OPS:=native_group_norm}"
: "${MUSETALK_TRT_STAGEWISE_TORCH_STAGES:=}"
: "${MUSETALK_TRT_STAGEWISE_PRECISION:=fp16}"
: "${MUSETALK_TRT_STAGEWISE_INT8_STAGES:=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2}"
: "${MUSETALK_TRT_STAGEWISE_WORKSPACE_GB:=2}"
: "${MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR:=./calibration/vae_decoder}"
: "${MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_BATCHES:=8}"
: "${MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO:=minmax}"
: "${MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR:=./models/tensorrt/stagewise_int8_onnx_qdq_cache}"
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
: "${MUSETALK_UNET_CALIBRATION_CAPTURE:=0}"
: "${MUSETALK_UNET_CALIBRATION_DIR:=./calibration/unet}"
: "${MUSETALK_UNET_CALIBRATION_MAX_BATCHES:=128}"

: "${WEBRTC_SYNC_MODE:=strict_fifo}"
: "${WEBRTC_VIDEO_PREBUFFER_SECONDS:=2.0}"
: "${WEBRTC_AUDIO_PREBUFFER_SECONDS:=0.0}"
: "${WEBRTC_ADAPTIVE_FPS:=0}"
: "${WEBRTC_BATCH_FRAME_CALLBACK:=1}"
: "${WEBRTC_POSE_CROSSFADE_FRAMES:=2}"
: "${MUSETALK_BLEND_FIXED_POINT:=1}"
: "${MUSETALK_BLEND_SHRINK_MASK_BBOX:=1}"

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

validate_int8_startup_requirements() {
  case "${MUSETALK_TRT_STAGEWISE_PRECISION,,}" in
    int8|int8_mixed|mixed_int8)
      ;;
    *)
      return 0
      ;;
  esac

  [[ -n "$MUSETALK_TRT_STAGEWISE_INT8_STAGES" ]] || die \
    "MUSETALK_TRT_STAGEWISE_PRECISION=$MUSETALK_TRT_STAGEWISE_PRECISION requires MUSETALK_TRT_STAGEWISE_INT8_STAGES"

  local calibration_dir="$MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR"
  if [[ "$calibration_dir" != /* ]]; then
    calibration_dir="$REPO_ROOT/$calibration_dir"
  fi
  [[ -d "$calibration_dir" ]] || die \
    "INT8 calibration directory not found: $calibration_dir"
  [[ -n "$(find "$calibration_dir" -type f -name '*.pt' -print -quit)" ]] || die \
    "INT8 calibration directory has no .pt captures: $calibration_dir"

  (
    cd "$REPO_ROOT"
    "$VENV_PYTHON" - <<'PY'
import modelopt.torch.quantization as mtq
import onnx
import tensorrt

if not hasattr(mtq, "INT8_DEFAULT_CFG"):
    raise RuntimeError("modelopt.torch.quantization.INT8_DEFAULT_CFG is unavailable")
PY
  ) || die \
    "INT8 startup requires nvidia-modelopt, onnx, and TensorRT in $VENV_PATH. Re-run setup_musetalk.sh or set MUSETALK_TRT_STAGEWISE_PRECISION=fp16 to opt out."
}

unet_trt_requested() {
  local requested=0
  case "${MUSETALK_UNET_BACKEND,,}" in
    trt|tensorrt)
      requested=1
      ;;
  esac
  case "${MUSETALK_TRT_UNET_ENABLED,,}" in
    1|true|yes|on)
      requested=1
      ;;
  esac
  (( requested ))
}

resolve_repo_path() {
  local raw="$1"
  if [[ "$raw" == /* ]]; then
    printf '%s\n' "$raw"
  else
    printf '%s/%s\n' "$REPO_ROOT" "$raw"
  fi
}

build_unet_trt_split8_if_needed() {
  unet_trt_requested || return 0
  case "${MUSETALK_TRT_UNET_BUILD,,}" in
    1|true|yes|on)
      ;;
    *)
      return 0
      ;;
  esac

  local output_dir engine_path meta_path capture_dir report_path
  output_dir="$(resolve_repo_path "$MUSETALK_TRT_UNET_OUTPUT_DIR")"
  engine_path="$output_dir/unet_trt.ts"
  meta_path="$output_dir/unet_trt_meta.json"
  if [[ -f "$engine_path" && -f "$meta_path" ]]; then
    return 0
  fi

  capture_dir="$(resolve_repo_path "$MUSETALK_TRT_UNET_CAPTURE_DIR")"
  [[ -d "$capture_dir" ]] || die \
    "TRT UNet split8 build requested, but capture directory is missing: $capture_dir"
  [[ -n "$(find "$capture_dir" -type f -name 'unet_io_*.pt' -print -quit)" ]] || die \
    "TRT UNet split8 build requested, but no unet_io_*.pt captures were found in: $capture_dir"

  report_path="$(resolve_repo_path "$MUSETALK_TRT_UNET_VALIDATE_REPORT_PATH")"
  mkdir -p "$output_dir" "$(dirname "$report_path")"

  log "Building missing TRT UNet split8 artifact"
  log "MUSETALK_TRT_UNET_OUTPUT_DIR=$output_dir"
  log "MUSETALK_TRT_UNET_CAPTURE_DIR=$capture_dir"
  (
    cd "$REPO_ROOT"
    "$VENV_PYTHON" scripts/tensorrt_export.py \
      --components unet \
      --batch-sizes "$MUSETALK_TRT_UNET_BUILD_BATCH_SIZES" \
      --output-dir "$output_dir" \
      --precision fp16 \
      --save-format exported_program \
      --workspace-gb "$MUSETALK_TRT_UNET_WORKSPACE_GB" \
      --min-block-size "$MUSETALK_TRT_UNET_MIN_BLOCK_SIZE" \
      --unet-capture-dir "$capture_dir" \
      --validate-unet-capture-dir "$capture_dir" \
      --validate-unet-limit "$MUSETALK_TRT_UNET_VALIDATE_LIMIT" \
      --validate-unet-padded-batch-size "$MUSETALK_TRT_UNET_VALIDATE_PADDED_BATCH_SIZE" \
      --validate-unet-report-path "$report_path" \
      --require-valid-unet
  ) || die "TRT UNet split8 build failed; see logs above and report path $report_path"

  [[ -f "$engine_path" && -f "$meta_path" ]] || die \
    "TRT UNet split8 build completed without expected artifact: $engine_path"
}

validate_unet_trt_startup_requirements() {
  unet_trt_requested || return 0

  (
    cd "$REPO_ROOT"
    REPO_ROOT="$REPO_ROOT" \
    MUSETALK_TRT_UNET_PATHS="$MUSETALK_TRT_UNET_PATHS" \
    MUSETALK_TRT_UNET_PATH="${MUSETALK_TRT_UNET_PATH:-}" \
    "$VENV_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
raw_paths = os.getenv("MUSETALK_TRT_UNET_PATHS", "").strip()
if not raw_paths:
    raw_paths = "8:" + os.getenv("MUSETALK_TRT_UNET_PATH", "./models/tensorrt/unet_trt.ts")

resolved: dict[int, Path] = {}
for token in raw_paths.split(","):
    token = token.strip()
    if not token:
        continue
    if ":" not in token:
        raise RuntimeError(
            "Invalid MUSETALK_TRT_UNET_PATHS entry "
            f"{token!r}; expected '<batch>:<path>'."
        )
    batch_raw, path_raw = token.split(":", 1)
    batch = int(batch_raw.strip())
    path = Path(path_raw.strip())
    if not path.is_absolute():
        path = repo / path
    resolved[batch] = path

if not resolved:
    raise RuntimeError("MUSETALK_TRT_UNET_PATHS did not resolve any batch paths")

allowed_batches = set()
for raw in os.getenv("HLS_SCHEDULER_FIXED_BATCH_SIZES", "").split(","):
    raw = raw.strip()
    if raw:
        allowed_batches.add(int(raw))
if not allowed_batches:
    for raw in os.getenv("MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES", "").split(","):
        raw = raw.strip()
        if raw:
            allowed_batches.add(int(raw))
if not allowed_batches:
    allowed_batches = set(resolved)

missing_batches = sorted(batch for batch in allowed_batches if batch not in resolved)
if missing_batches:
    raise RuntimeError(
        "TRT UNet startup is missing artifacts for configured scheduler batch "
        f"sizes: {missing_batches}. Resolved batches: {sorted(resolved)}"
    )

preferred_batch = max(allowed_batches)
engine_path = resolved[preferred_batch]
if not engine_path.exists():
    raise FileNotFoundError(f"TensorRT UNet engine not found: {engine_path}")

meta_path = engine_path.with_name("unet_trt_meta.json")
if not meta_path.exists():
    raise FileNotFoundError(f"TensorRT UNet metadata not found: {meta_path}")

meta = json.loads(meta_path.read_text())
validation = meta.get("validation")
if validation and validation.get("passed") is False:
    raise RuntimeError(f"TensorRT UNet artifact failed validation: {meta_path}")
PY
  ) || die \
    "TRT UNet startup requires validated artifacts matching MUSETALK_TRT_UNET_PATHS and scheduler batches. Restore the TRT artifact bundle or set MUSETALK_TRT_UNET_ENABLED=0 and MUSETALK_UNET_BACKEND= to opt out."
}

case "$PROFILE" in
  baseline|throughput_record|vram_max)
    apply_gpu_aware_defaults "$PROFILE"
    ;;
  *)
    die "Unsupported profile: $PROFILE"
    ;;
esac

validate_int8_startup_requirements
build_unet_trt_split8_if_needed
validate_unet_trt_startup_requirements

export MUSETALK_COMPILE
export MUSETALK_COMPILE_UNET
export MUSETALK_COMPILE_VAE
export MUSETALK_WARM_RUNTIME
export MUSETALK_TRT_ENABLED
export MUSETALK_VAE_BACKEND
export MUSETALK_TRT_FALLBACK
export MUSETALK_TRT_PROFILE_NAME
export MUSETALK_UNET_BACKEND
export MUSETALK_TRT_UNET_ENABLED
export MUSETALK_TRT_UNET_PATHS
export MUSETALK_TRT_UNET_PATH
export MUSETALK_TRT_UNET_META_PATH
export MUSETALK_TRT_UNET_ALLOW_UNVALIDATED
export MUSETALK_TRT_STAGEWISE_TORCH_EXECUTED_OPS
export MUSETALK_TRT_STAGEWISE_TORCH_STAGES
export MUSETALK_TRT_STAGEWISE_PRECISION
export MUSETALK_TRT_STAGEWISE_WORKSPACE_GB
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
export MUSETALK_TRT_UNET_BUILD
export MUSETALK_TRT_UNET_OUTPUT_DIR
export MUSETALK_TRT_UNET_CAPTURE_DIR
export MUSETALK_TRT_UNET_BUILD_BATCH_SIZES
export MUSETALK_TRT_UNET_WORKSPACE_GB
export MUSETALK_TRT_UNET_MIN_BLOCK_SIZE
export MUSETALK_TRT_UNET_VALIDATE_LIMIT
export MUSETALK_TRT_UNET_VALIDATE_PADDED_BATCH_SIZE
export MUSETALK_TRT_UNET_VALIDATE_REPORT_PATH
export MUSETALK_UNET_CALIBRATION_CAPTURE
export MUSETALK_UNET_CALIBRATION_DIR
export MUSETALK_UNET_CALIBRATION_MAX_BATCHES
export WEBRTC_SYNC_MODE
export WEBRTC_VIDEO_PREBUFFER_SECONDS
export WEBRTC_AUDIO_PREBUFFER_SECONDS
export WEBRTC_ADAPTIVE_FPS
export WEBRTC_BATCH_FRAME_CALLBACK
export WEBRTC_POSE_CROSSFADE_FRAMES
export MUSETALK_BLEND_FIXED_POINT
export MUSETALK_BLEND_SHRINK_MASK_BBOX
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
if [[ -n "$MUSETALK_TRT_PROFILE_NAME" ]]; then
  log "MUSETALK_TRT_PROFILE_NAME=$MUSETALK_TRT_PROFILE_NAME"
fi
log "GPU_TOTAL_MEMORY_GB=$GPU_TOTAL_MEMORY_GB"
log "GPU_RESERVED_MEMORY_GB=$GPU_RESERVED_MEMORY_GB"
log "GPU_MEMORY_DETECTION_SOURCE=${GPU_MEMORY_DETECTION_SOURCE:-unknown}"
log "HLS_SCHEDULER_MAX_BATCH=$HLS_SCHEDULER_MAX_BATCH"
log "HLS_SCHEDULER_FIXED_BATCH_SIZES=$HLS_SCHEDULER_FIXED_BATCH_SIZES"
log "MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=$MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES"
log "MUSETALK_TRT_STAGEWISE_PRECISION=$MUSETALK_TRT_STAGEWISE_PRECISION"
log "MUSETALK_TRT_STAGEWISE_WORKSPACE_GB=$MUSETALK_TRT_STAGEWISE_WORKSPACE_GB"
log "MUSETALK_TRT_STAGEWISE_INT8_STAGES=${MUSETALK_TRT_STAGEWISE_INT8_STAGES:-default}"
log "MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR=${MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR:-${MUSETALK_VAE_CALIBRATION_DIR}}"
log "MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO=$MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO"
log "MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR=$MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR"
log "MUSETALK_VAE_CALIBRATION_CAPTURE=$MUSETALK_VAE_CALIBRATION_CAPTURE"
log "MUSETALK_UNET_BACKEND=$MUSETALK_UNET_BACKEND"
log "MUSETALK_TRT_UNET_ENABLED=$MUSETALK_TRT_UNET_ENABLED"
log "MUSETALK_TRT_UNET_PATHS=$MUSETALK_TRT_UNET_PATHS"
log "MUSETALK_TRT_UNET_BUILD=$MUSETALK_TRT_UNET_BUILD"
log "MUSETALK_TRT_UNET_CAPTURE_DIR=$MUSETALK_TRT_UNET_CAPTURE_DIR"
log "MUSETALK_UNET_CALIBRATION_CAPTURE=$MUSETALK_UNET_CALIBRATION_CAPTURE"
log "WEBRTC_BATCH_FRAME_CALLBACK=$WEBRTC_BATCH_FRAME_CALLBACK"
log "WEBRTC_POSE_CROSSFADE_FRAMES=$WEBRTC_POSE_CROSSFADE_FRAMES"
log "MUSETALK_BLEND_FIXED_POINT=$MUSETALK_BLEND_FIXED_POINT"
log "MUSETALK_BLEND_SHRINK_MASK_BBOX=$MUSETALK_BLEND_SHRINK_MASK_BBOX"
log "WEBRTC_H264_ENCODER=${WEBRTC_H264_ENCODER:-h264_nvenc(default)}"
log "WEBRTC_ICE_TRANSPORT_POLICY=${WEBRTC_ICE_TRANSPORT_POLICY:-all(default)}"
log "WEBRTC_STUN_URLS=${WEBRTC_STUN_URLS:-stun:stun.l.google.com:19302(default)}"
log "WEBRTC_TURN_URLS=${WEBRTC_TURN_URLS:-}"
log "WEBRTC_SERVER_TURN_URLS=${WEBRTC_SERVER_TURN_URLS:-}"
log "WEBRTC_SYNC_MODE=$WEBRTC_SYNC_MODE"
log "WEBRTC_VIDEO_PREBUFFER_SECONDS=$WEBRTC_VIDEO_PREBUFFER_SECONDS"
log "WEBRTC_ADAPTIVE_FPS=$WEBRTC_ADAPTIVE_FPS"
log "PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER PYTHONUNBUFFERED=$PYTHONUNBUFFERED"

if (( VALIDATE_ONLY )); then
  log "Validation-only checks passed; API server was not started"
  exit 0
fi

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
