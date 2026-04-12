#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-models}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-1}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
DOWNLOAD_RETRY_SLEEP_SECONDS="${DOWNLOAD_RETRY_SLEEP_SECONDS:-15}"
DOWNLOAD_WAIT_FOR_NETWORK_SECONDS="${DOWNLOAD_WAIT_FOR_NETWORK_SECONDS:-180}"
DOWNLOAD_WAIT_FOR_NETWORK_INTERVAL_SECONDS="${DOWNLOAD_WAIT_FOR_NETWORK_INTERVAL_SECONDS:-5}"
DOWNLOAD_AVATAR_PREP_WEIGHTS="${DOWNLOAD_AVATAR_PREP_WEIGHTS:-1}"
EXPECTED_FACE_PARSE_ITER_BYTES="${EXPECTED_FACE_PARSE_ITER_BYTES:-53289463}"
EXPECTED_RESNET18_BYTES="${EXPECTED_RESNET18_BYTES:-46827520}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

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

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

retry() {
  local attempt=1
  local status=0

  while true; do
    if "$@"; then
      return 0
    else
      status=$?
    fi

    if (( attempt >= DOWNLOAD_RETRIES )); then
      return "$status"
    fi

    log "Attempt ${attempt}/${DOWNLOAD_RETRIES} failed with exit ${status}; retrying in ${DOWNLOAD_RETRY_SLEEP_SECONDS}s: $*"
    sleep "$DOWNLOAD_RETRY_SLEEP_SECONDS"
    attempt=$((attempt + 1))
  done
}

wait_for_http_endpoint() {
  local url="$1"
  local timeout_seconds="$2"
  local interval_seconds="$3"
  local start_ts
  local elapsed=0

  start_ts="$(date +%s)"
  while true; do
    if curl --silent --show-error --fail --location --connect-timeout 5 --max-time 20 \
      --head "$url" >/dev/null; then
      return 0
    fi

    elapsed=$(( $(date +%s) - start_ts ))
    if (( elapsed >= timeout_seconds )); then
      return 1
    fi

    log "Waiting for outbound HTTPS to become ready (${elapsed}s/${timeout_seconds}s): $url"
    sleep "$interval_seconds"
  done
}

hf_download() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2

  retry huggingface-cli download \
    --max-workers "$HF_MAX_WORKERS" \
    --local-dir "$local_dir" \
    "$repo_id" \
    "$@"
}

remote_content_length() {
  local url="$1"
  curl -fsSI "$url" | tr -d '\r' | awk '/^content-length:/ {print $2; exit}'
}

download_resumable_with_curl() {
  local url="$1"
  local output_path="$2"
  local local_size=0
  local remote_size

  remote_size="$(remote_content_length "$url")"
  [[ -n "$remote_size" ]] || die "Could not determine remote content length for $url"

  if [[ -f "$output_path" ]]; then
    local_size="$(stat -c '%s' "$output_path")"
    if [[ "$local_size" == "$remote_size" ]]; then
      log "File already complete, skipping: $output_path"
      return 0
    fi
    if (( local_size > remote_size )); then
      log "Local file is larger than remote; restarting download: $output_path"
      rm -f "$output_path"
    fi
  fi

  retry curl --fail --location --retry 5 --retry-delay 5 --retry-all-errors --continue-at - \
    "$url" \
    -o "$output_path"
}

download_gdown_if_needed() {
  local url="$1"
  local output_path="$2"
  local expected_bytes="$3"
  local local_size=0

  if [[ -f "$output_path" ]]; then
    local_size="$(stat -c '%s' "$output_path")"
    if [[ "$local_size" == "$expected_bytes" ]]; then
      log "File already complete, skipping: $output_path"
      return 0
    fi
    if (( local_size > expected_bytes )); then
      log "Local file is larger than expected; restarting download: $output_path"
      rm -f "$output_path"
    fi
  fi

  retry gdown "$url" -O "$output_path"
}

validate_required_files() {
  local required=(
    "$CHECKPOINTS_DIR/musetalk/musetalk.json"
    "$CHECKPOINTS_DIR/musetalk/pytorch_model.bin"
    "$CHECKPOINTS_DIR/musetalkV15/musetalk.json"
    "$CHECKPOINTS_DIR/musetalkV15/unet.pth"
    "$CHECKPOINTS_DIR/sd-vae/config.json"
    "$CHECKPOINTS_DIR/sd-vae/diffusion_pytorch_model.bin"
    "$CHECKPOINTS_DIR/whisper/config.json"
    "$CHECKPOINTS_DIR/whisper/pytorch_model.bin"
    "$CHECKPOINTS_DIR/whisper/preprocessor_config.json"
    "$CHECKPOINTS_DIR/face-parse-bisent/79999_iter.pth"
    "$CHECKPOINTS_DIR/face-parse-bisent/resnet18-5c106cde.pth"
  )

  if env_flag_is_true "$DOWNLOAD_AVATAR_PREP_WEIGHTS"; then
    required+=(
      "$CHECKPOINTS_DIR/dwpose/dw-ll_ucoco_384.pth"
      "$CHECKPOINTS_DIR/syncnet/latentsync_syncnet.pt"
      "$CHECKPOINTS_DIR/auxiliary/s3fd-619a316812.pth"
      "$CHECKPOINTS_DIR/face_detection/s3fd.pth"
    )
  fi

  local missing=()
  local path
  for path in "${required[@]}"; do
    if [[ ! -s "$path" ]]; then
      missing+=("$path")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    printf '[%s] ERROR: Missing required model files:\n' "$SCRIPT_NAME" >&2
    printf '  %s\n' "${missing[@]}" >&2
    return 1
  fi
}

require_command huggingface-cli
require_command curl
require_command python

log "Using Hugging Face endpoint: $HF_ENDPOINT"
log "Hugging Face timeouts: etag=${HF_HUB_ETAG_TIMEOUT}s download=${HF_HUB_DOWNLOAD_TIMEOUT}s"
log "Hugging Face max workers: $HF_MAX_WORKERS"
log "Download retries: $DOWNLOAD_RETRIES"
log "Avatar-prep weights enabled: $DOWNLOAD_AVATAR_PREP_WEIGHTS"

# Create necessary directories
mkdir -p \
  "$CHECKPOINTS_DIR/musetalk" \
  "$CHECKPOINTS_DIR/musetalkV15" \
  "$CHECKPOINTS_DIR/syncnet" \
  "$CHECKPOINTS_DIR/dwpose" \
  "$CHECKPOINTS_DIR/face-parse-bisent" \
  "$CHECKPOINTS_DIR/sd-vae" \
  "$CHECKPOINTS_DIR/whisper" \
  "$CHECKPOINTS_DIR/auxiliary" \
  "$CHECKPOINTS_DIR/face_detection"

if ! wait_for_http_endpoint "$HF_ENDPOINT" \
  "$DOWNLOAD_WAIT_FOR_NETWORK_SECONDS" \
  "$DOWNLOAD_WAIT_FOR_NETWORK_INTERVAL_SECONDS"; then
  die "Outbound HTTPS never became ready for $HF_ENDPOINT within ${DOWNLOAD_WAIT_FOR_NETWORK_SECONDS}s"
fi

log "Ensuring download helpers are installed"
retry python -m pip install --disable-pip-version-check --no-cache-dir gdown

log "Downloading MuseTalk V1.0 weights"
hf_download TMElyralab/MuseTalk "$CHECKPOINTS_DIR" \
  musetalk/musetalk.json \
  musetalk/pytorch_model.bin

log "Downloading MuseTalk V1.5 weights"
hf_download TMElyralab/MuseTalk "$CHECKPOINTS_DIR" \
  musetalkV15/musetalk.json \
  musetalkV15/unet.pth

log "Downloading SD VAE weights"
hf_download stabilityai/sd-vae-ft-mse "$CHECKPOINTS_DIR/sd-vae" \
  config.json \
  diffusion_pytorch_model.bin

log "Downloading Whisper weights"
hf_download openai/whisper-tiny "$CHECKPOINTS_DIR/whisper" \
  config.json \
  pytorch_model.bin \
  preprocessor_config.json

if env_flag_is_true "$DOWNLOAD_AVATAR_PREP_WEIGHTS"; then
  log "Downloading DWPose weights"
  hf_download yzd-v/DWPose "$CHECKPOINTS_DIR/dwpose" \
    dw-ll_ucoco_384.pth

  log "Downloading SyncNet weights"
  hf_download ByteDance/LatentSync "$CHECKPOINTS_DIR/syncnet" \
    latentsync_syncnet.pt

  log "Downloading S3FD face detector weights"
  hf_download ByteDance/LatentSync "$CHECKPOINTS_DIR" \
    auxiliary/s3fd-619a316812.pth
  [[ -s "$CHECKPOINTS_DIR/auxiliary/s3fd-619a316812.pth" ]] || \
    die "S3FD download step completed without producing $CHECKPOINTS_DIR/auxiliary/s3fd-619a316812.pth"
  cp -f "$CHECKPOINTS_DIR/auxiliary/s3fd-619a316812.pth" "$CHECKPOINTS_DIR/face_detection/s3fd.pth"
else
  log "Skipping avatar-prep-only weights (dwpose, syncnet, s3fd)"
fi

log "Downloading face parse model"
download_gdown_if_needed \
  "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812" \
  "$CHECKPOINTS_DIR/face-parse-bisent/79999_iter.pth" \
  "$EXPECTED_FACE_PARSE_ITER_BYTES"

log "Downloading ResNet18 backbone"
download_resumable_with_curl \
  "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
  "$CHECKPOINTS_DIR/face-parse-bisent/resnet18-5c106cde.pth"

validate_required_files
log "All weights downloaded and validated successfully"
