#!/usr/bin/env bash
set -euo pipefail

# ── Logging to file ──────────────────────────────────────────────────────────
ONSTART_LOG="${ONSTART_LOG:-/workspace/onstart.log}"
ONSTART_START_TS="$(date +%s)"
ONSTART_START_UTC="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
exec > >(tee -a "$ONSTART_LOG") 2>&1
echo ""
echo "========================================"
echo "VAST_ONSTART BEGIN: $ONSTART_START_UTC"
echo "========================================"

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE:-/workspace}"
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/.venvs/musetalk_trt_stagewise}"
PROFILE="${PROFILE:-throughput_record}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
AUTO_SETUP="${AUTO_SETUP:-1}"
SETUP_CLEAN="${SETUP_CLEAN:-0}"
SETUP_SKIP_APT="${SETUP_SKIP_APT:-auto}"
SETUP_SKIP_WEIGHTS="${SETUP_SKIP_WEIGHTS:-0}"
SETUP_FULL_STACK="${SETUP_FULL_STACK:-0}"
SETUP_INSTALL_AVATAR_PREP_DEPS="${SETUP_INSTALL_AVATAR_PREP_DEPS:-0}"
SETUP_WEBRTC_TURN="${SETUP_WEBRTC_TURN:-auto}"
TURN_ENV_FILE="${TURN_ENV_FILE:-$REPO_ROOT/.env.webrtc-turn.local}"
TURN_ENV_FORCE="${TURN_ENV_FORCE:-0}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"

log() {
  printf '[%s] [%s] %s\n' "$SCRIPT_NAME" "$(date -u '+%H:%M:%S')" "$*"
}

format_duration() {
  local total_seconds="${1:-0}"
  local minutes=$((total_seconds / 60))
  local seconds=$((total_seconds % 60))

  if (( minutes > 0 )); then
    printf '%dm%02ds' "$minutes" "$seconds"
  else
    printf '%ss' "$seconds"
  fi
}

elapsed_since_start() {
  printf '%s\n' "$(( $(date +%s) - ONSTART_START_TS ))"
}

die() {
  printf '[%s] [%s] ERROR: %s\n' "$SCRIPT_NAME" "$(date -u '+%H:%M:%S')" "$*" >&2
  echo "========================================"
  echo "VAST_ONSTART FAILED: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "TOTAL ELAPSED: $(format_duration "$(elapsed_since_start)")"
  echo "LOG FILE: $ONSTART_LOG"
  echo "========================================"
  trap - ERR
  exit 1
}

report_unhandled_failure() {
  local status=$?
  local line="${BASH_LINENO[0]:-${LINENO:-unknown}}"
  trap - ERR
  printf '[%s] [%s] ERROR: unhandled failure near line %s (exit %s)\n' \
    "$SCRIPT_NAME" "$(date -u '+%H:%M:%S')" "$line" "$status" >&2
  echo "========================================"
  echo "VAST_ONSTART FAILED: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "TOTAL ELAPSED: $(format_duration "$(elapsed_since_start)")"
  echo "LOG FILE: $ONSTART_LOG"
  echo "========================================"
  exit "$status"
}

trap report_unhandled_failure ERR

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

webrtc_turn_requested() {
  case "${SETUP_WEBRTC_TURN,,}" in
    1|true|yes|on|auto)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

webrtc_turn_required() {
  case "${SETUP_WEBRTC_TURN,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

read_proc1_env() {
  local key="$1"
  if [[ -r /proc/1/environ ]]; then
    tr '\0' '\n' < /proc/1/environ 2>/dev/null | awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}'
  fi
}

generate_turn_password() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 24
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import secrets
print(secrets.token_hex(24))
PY
    return 0
  fi
  date +%s%N
}

ensure_coturn_available() {
  if command -v turnserver >/dev/null 2>&1; then
    return 0
  fi

  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    die "WebRTC TURN autostart requires coturn, but turnserver is not installed and this script is not running as root"
  fi

  log "Installing coturn for WebRTC TURN autostart"
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get install -y coturn
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
import boto3
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
    "$REPO_ROOT/models/face_detection/s3fd.pth"
  )

  local path
  for path in "${required[@]}"; do
    [[ -e "$path" ]] || return 1
  done

  avatar_prep_runtime_imports_complete || return 1
}

run_setup_if_needed() {
  export HF_MAX_WORKERS

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

  if avatar_prep_requested; then
    log "Full-stack bootstrap requested: this node will install avatar-prep deps and weights in addition to the TRT server runtime"
    log "Current torch 2.5.x + CUDA 12.1 full-stack boots may source-build mmcv when OpenMMLab does not publish a matching prebuilt wheel"
  else
    log "Using the faster server-only bootstrap path; skipping optional avatar-prep deps and weights"
    log "Enable SETUP_FULL_STACK=1 only if this node must handle /avatars/prepare directly"
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

  log "Using HF_MAX_WORKERS=$HF_MAX_WORKERS for setup/download flow"
  log "Running setup_musetalk.sh ${setup_args[*]}"
  bash "$REPO_ROOT/setup_musetalk.sh" "${setup_args[@]}"
}

# ── Post-setup validation (logged) ──────────────────────────────────────────
run_post_setup_validation() {
  log "── Post-setup validation ──"
  local PY="$VENV_PATH/bin/python"
  local all_ok=true

  if [[ -x "$PY" ]]; then
    log "✅ Venv Python exists at $PY"
  else
    log "❌ Venv Python NOT found at $PY"
    all_ok=false
  fi

  # Check critical model files
  local model_files=(
    "models/musetalkV15/unet.pth"
    "models/sd-vae/diffusion_pytorch_model.bin"
    "models/whisper/pytorch_model.bin"
    "models/face-parse-bisent/79999_iter.pth"
  )

  if avatar_prep_requested; then
    model_files+=(
      "models/dwpose/dw-ll_ucoco_384.pth"
      "models/syncnet/latentsync_syncnet.pt"
      "models/face_detection/s3fd.pth"
    )
  fi

  for f in "${model_files[@]}"; do
    if [[ -f "$REPO_ROOT/$f" ]]; then
      local size
      size=$(du -h "$REPO_ROOT/$f" | cut -f1)
      log "✅ $f ($size)"
    else
      log "❌ MISSING: $f"
      all_ok=false
    fi
  done

  # Check Python imports
  if [[ -x "$PY" ]]; then
    if (cd "$REPO_ROOT" && $PY -c "import torch; print(f'torch {torch.__version__}, CUDA={torch.cuda.is_available()}')" 2>&1); then
      log "✅ torch + CUDA OK"
    else
      log "❌ torch import failed"
      all_ok=false
    fi

    if (cd "$REPO_ROOT" && $PY -c "import boto3, uvicorn, fastapi; print('boto3 + uvicorn + fastapi OK')" 2>&1); then
      log "✅ Server deps OK"
    else
      log "❌ Server deps missing"
      all_ok=false
    fi

    if avatar_prep_requested; then
      if (cd "$REPO_ROOT" && $PY -c "import mmcv, mmdet, mmpose; print('mmcv + mmdet + mmpose OK')" 2>&1); then
        log "✅ Avatar prep deps OK"
      else
        log "❌ Avatar prep deps missing (mmcv/mmdet/mmpose)"
        all_ok=false
      fi
    fi
  fi

  if $all_ok; then
    log "✅ All post-setup validation checks passed"
  else
    log "⚠️  Some validation checks failed — check log above"
  fi
}

bootstrap_runtime_secrets() {
  local secret_id="${MUSETALK_AWS_SECRET_ID:-}"
  if [[ -z "$secret_id" ]]; then
    log "MuseTalk AWS Secrets Manager bootstrap skipped (MUSETALK_AWS_SECRET_ID not set)"
    return 0
  fi

  local PY="$VENV_PATH/bin/python"
  [[ -x "$PY" ]] || die "Cannot bootstrap runtime secrets; venv Python not found at $PY"

  local strict="${MUSETALK_SECRETS_STRICT:-${SECRETS_STRICT:-true}}"
  local verify_s3="${MUSETALK_SECRETS_VERIFY_S3:-1}"
  local tmp_env
  tmp_env="$(mktemp "$WORKSPACE_ROOT/.musetalk-runtime-secret.XXXXXX.env")"
  chmod 600 "$tmp_env"

  log "Bootstrapping MuseTalk runtime env from AWS Secrets Manager"
  if env_flag_is_true "$verify_s3"; then
    log "Secret bootstrap S3 verification is enabled"
  else
    log "Secret bootstrap S3 verification is disabled"
  fi

  local bootstrap_args=("--output" "$tmp_env")
  if env_flag_is_true "$verify_s3"; then
    bootstrap_args+=("--verify-s3")
  fi

  if (
    cd "$REPO_ROOT"
    "$PY" "$REPO_ROOT/scripts/bootstrap_aws_secrets.py" "${bootstrap_args[@]}"
  ); then
    # shellcheck disable=SC1090
    source "$tmp_env"
    rm -f "$tmp_env"
    log "MuseTalk runtime secret env exports loaded"
    return 0
  fi

  rm -f "$tmp_env"
  if env_flag_is_true "$strict"; then
    die "AWS Secrets Manager bootstrap failed and strict mode is enabled"
  fi
  log "⚠️  AWS Secrets Manager bootstrap failed; continuing because strict mode is disabled"
}

configure_webrtc_turn() {
  if ! webrtc_turn_requested; then
    log "WebRTC TURN bootstrap disabled (SETUP_WEBRTC_TURN=$SETUP_WEBRTC_TURN)"
    return 0
  fi

  if [[ -f "$TURN_ENV_FILE" ]] && ! env_flag_is_true "$TURN_ENV_FORCE"; then
    set -a
    # shellcheck disable=SC1090
    source "$TURN_ENV_FILE"
    set +a
    export TURN_ENV_FILE WEBRTC_RELAY_ENABLED WEBRTC_TURN_AUTOSTART
    if env_flag_is_true "${WEBRTC_TURN_AUTOSTART:-0}"; then
      ensure_coturn_available
    fi
    log "Loaded existing WebRTC TURN env from $TURN_ENV_FILE"
    return 0
  fi

  local public_ip vast_tcp_1455 vast_udp_3478 listen_port public_port transport turn_pass
  public_ip="${TURN_PUBLIC_IP:-${PUBLIC_IPADDR:-$(read_proc1_env PUBLIC_IPADDR)}}"
  vast_tcp_1455="${VAST_TCP_PORT_1455:-$(read_proc1_env VAST_TCP_PORT_1455)}"
  vast_udp_3478="${VAST_UDP_PORT_3478:-$(read_proc1_env VAST_UDP_PORT_3478)}"

  if [[ -z "$public_ip" ]]; then
    if webrtc_turn_required; then
      die "SETUP_WEBRTC_TURN=$SETUP_WEBRTC_TURN but no TURN_PUBLIC_IP/PUBLIC_IPADDR could be detected"
    fi
    log "WebRTC TURN auto bootstrap skipped: no public IP detected"
    return 0
  fi

  if [[ -n "$vast_tcp_1455" ]]; then
    listen_port="${TURN_LISTEN_PORT:-1455}"
    public_port="${TURN_PUBLIC_PORT:-$vast_tcp_1455}"
    transport="${TURN_PUBLIC_TRANSPORT:-tcp}"
  elif [[ -n "$vast_udp_3478" ]]; then
    listen_port="${TURN_LISTEN_PORT:-3478}"
    public_port="${TURN_PUBLIC_PORT:-$vast_udp_3478}"
    transport="${TURN_PUBLIC_TRANSPORT:-udp}"
  else
    if webrtc_turn_required; then
      die "SETUP_WEBRTC_TURN=$SETUP_WEBRTC_TURN but no Vast TURN port mapping was detected (expected VAST_TCP_PORT_1455 or VAST_UDP_PORT_3478)"
    fi
    log "WebRTC TURN auto bootstrap skipped: no Vast TURN port mapping detected"
    return 0
  fi

  turn_pass="${TURN_PASS:-${WEBRTC_TURN_PASS:-$(generate_turn_password)}}"
  mkdir -p "$(dirname "$TURN_ENV_FILE")"
  umask 077
  cat > "$TURN_ENV_FILE" <<EOF
WEBRTC_RELAY_ENABLED=1
WEBRTC_TURN_AUTOSTART=1

TURN_PUBLIC_IP=$public_ip
TURN_PUBLIC_PORT=$public_port
TURN_PUBLIC_TRANSPORT=$transport
TURN_LISTEN_PORT=$listen_port
WEBRTC_USE_LOCAL_TURN=1

TURN_USER=${TURN_USER:-webrtc}
TURN_PASS=$turn_pass

WEBRTC_ICE_TRANSPORT_POLICY=relay
WEBRTC_STUN_URLS=
WEBRTC_SYNC_MODE=strict_fifo
WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0
WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0
WEBRTC_ADAPTIVE_FPS=0
EOF
  chmod 600 "$TURN_ENV_FILE"

  set -a
  # shellcheck disable=SC1090
  source "$TURN_ENV_FILE"
  set +a
  export TURN_ENV_FILE WEBRTC_RELAY_ENABLED WEBRTC_TURN_AUTOSTART
  ensure_coturn_available
  log "Generated WebRTC TURN env at $TURN_ENV_FILE"
  log "WebRTC TURN public URL: turn:$TURN_PUBLIC_IP:$TURN_PUBLIC_PORT?transport=$TURN_PUBLIC_TRANSPORT"
}

main() {
  local phase_start phase_elapsed total_elapsed
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
  log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
  log "Disk free: $(df -h /workspace | tail -1 | awk '{print $4}')"

  phase_start="$(date +%s)"
  run_setup_if_needed
  phase_elapsed="$(( $(date +%s) - phase_start ))"
  log "Bootstrap/setup phase finished in $(format_duration "$phase_elapsed")"

  phase_start="$(date +%s)"
  run_post_setup_validation
  phase_elapsed="$(( $(date +%s) - phase_start ))"
  log "Post-setup validation finished in $(format_duration "$phase_elapsed")"

  phase_start="$(date +%s)"
  bootstrap_runtime_secrets
  phase_elapsed="$(( $(date +%s) - phase_start ))"
  log "Runtime secret bootstrap phase finished in $(format_duration "$phase_elapsed")"

  phase_start="$(date +%s)"
  configure_webrtc_turn
  phase_elapsed="$(( $(date +%s) - phase_start ))"
  log "WebRTC TURN bootstrap phase finished in $(format_duration "$phase_elapsed")"

  phase_start="$(date +%s)"
  PROFILE="$PROFILE" \
  HOST="$HOST" \
  PORT="$PORT" \
  REPO_ROOT="$REPO_ROOT" \
  VENV_PATH="$VENV_PATH" \
  bash "$REPO_ROOT/scripts/vast_server_ctl.sh" start
  phase_elapsed="$(( $(date +%s) - phase_start ))"
  log "Server start-to-health phase finished in $(format_duration "$phase_elapsed")"

  total_elapsed="$(elapsed_since_start)"
  log "Overall on-start completed in $(format_duration "$total_elapsed")"
  log "Vast.ai MuseTalk on-start complete"

  echo "========================================"
  echo "VAST_ONSTART COMPLETE: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "TOTAL ELAPSED: $(format_duration "$total_elapsed")"
  echo "LOG FILE: $ONSTART_LOG"
  echo "========================================"
}

main "$@"
