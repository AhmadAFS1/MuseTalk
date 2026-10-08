#!/usr/bin/env bash
# Vast.ai / box on-start: install check -> secrets -> TURN -> [r5 bundle] -> engines -> server
# (docs/STARTUP.md).
# Recipe r5 (default): an r5 engine bundle from S3 is restored and verified (the first candidate of the bundle:<a>|<b>
# list in configs/recipes/r5.env that fits this host: the RTX 4070 SUPER bundle there, else the portable AMPERE_PLUS
# bundle, which runs on any Ampere-or-newer GPU such as an RTX 3090), then vast_server_ctl.sh start
# (run_musetalk_server.sh + recipe verification after /health). No .ts UNet is built. A GPU that fits no candidate
# (older than Ampere) serves eager UNet + compiled TAESD with r5's serving levers.
# Recipe fast/fast300: install_musetalk.sh --check, unet_engine_store.py ensure (.ts UNet), ctl start.
# Recipe legacy_int8: the old TRT artifact restore + profile selector + legacy launcher.
# Every exit path prints exactly one VAST_ONSTART COMPLETE or VAST_ONSTART FAILED marker.
set -Eeuo pipefail

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
ONSTART_MAIN_PID="$$"
ONSTART_MARKER_PRINTED=0
WORKSPACE_ROOT="${WORKSPACE:-/workspace}"
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/.venvs/musetalk_trt_stagewise}"
PROFILE="${PROFILE:-throughput_record}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
AUTO_SETUP="${AUTO_SETUP:-1}"
SETUP_CLEAN="${SETUP_CLEAN:-0}"
# Opt-in image contract; snapshot it before a runtime secret can change env values.
# Non-container installs retain their existing repair/fallback behavior.
readonly IMMUTABLE_BOOT="${MUSETALK_IMMUTABLE_RUNTIME:-0}"
IMMUTABLE_SECRETS_LOADED=0
BOOTSTRAP_SECRET_TMP=""
if [[ "$IMMUTABLE_BOOT" == "1" ]]; then
  # An env secret may carry stale operational settings, but it must never replace
  # this supervisor's identity/PID paths.
  readonly MUSETALK_SUPERVISOR_OWNER PID_FILE TURN_PID_FILE MUSETALK_BOOTSTRAP_SECRET_DIR
fi
SETUP_SKIP_APT="${SETUP_SKIP_APT:-auto}"
SETUP_SKIP_WEIGHTS="${SETUP_SKIP_WEIGHTS:-0}"
SETUP_FULL_STACK="${SETUP_FULL_STACK:-0}"
SETUP_INSTALL_AVATAR_PREP_DEPS="${SETUP_INSTALL_AVATAR_PREP_DEPS:-0}"
SETUP_WEBRTC_TURN="${SETUP_WEBRTC_TURN:-auto}"
TURN_ENV_FILE="${TURN_ENV_FILE:-$REPO_ROOT/.env.webrtc-turn.local}"
TURN_ENV_FORCE="${TURN_ENV_FORCE:-0}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"
MUSETALK_SELECT_BEST_TRT_PROFILE="${MUSETALK_SELECT_BEST_TRT_PROFILE:-1}"
MUSETALK_TRT_PROFILE_ENV_FILE="${MUSETALK_TRT_PROFILE_ENV_FILE:-$REPO_ROOT/.runtime/musetalk_trt_best.env}"
MUSETALK_TRT_PROFILE_PREFER="${MUSETALK_TRT_PROFILE_PREFER:-split8}"
MUSETALK_TRT_ARTIFACT_RESTORE="${MUSETALK_TRT_ARTIFACT_RESTORE:-required}"
MUSETALK_TRT_ARTIFACT_STRICT="${MUSETALK_TRT_ARTIFACT_STRICT:-1}"
MUSETALK_TRT_ARTIFACT_KEY="${MUSETALK_TRT_ARTIFACT_KEY:-trt-artifacts/rtx3090/split8-int8/sha256-851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18/musetalk-trt-int8-split8.tar.gz}"
MUSETALK_TRT_ARTIFACT_SHA256="${MUSETALK_TRT_ARTIFACT_SHA256-851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18}"
# Recipe r5: the pinned engine bundles that configs/recipes/r5.env names (bundle:<a>|<b>, in order of preference;
# URI, SHA-256, host rule and sidecar dir live in configs/trt_bundles/<name>.json), so the boot restores exactly what
# the resolver will check. RESTORE: required (default; a host that fits a candidate but cannot restore any fails the
# boot), auto (warn and serve without it) or off. A host that fits no candidate (resolver bundle-check) skips it.
MUSETALK_R5_BUNDLE_RESTORE="${MUSETALK_R5_BUNDLE_RESTORE:-required}"
MUSETALK_TRT_ARTIFACT_STAGE_DIR="${MUSETALK_TRT_ARTIFACT_STAGE_DIR:-$REPO_ROOT/tmp/trt_artifact_stage}"
R5_BUNDLE_ACTIVE=0
INSTALLER="${MUSETALK_INSTALLER:-$REPO_ROOT/scripts/install_musetalk.sh}"
ENGINE_STORE="${MUSETALK_ENGINE_STORE_TOOL:-$REPO_ROOT/scripts/unet_engine_store.py}"
ONSTART_POST_VALIDATE_IMPORTS="${ONSTART_POST_VALIDATE_IMPORTS:-1}"
ONSTART_RECIPE="r5"
ONSTART_RECIPE_SOURCE="default"
# shellcheck source=lib/musetalk_env_layers.sh
MT_ENV_LOG_PREFIX="$SCRIPT_NAME"
source "$SCRIPT_DIR/lib/musetalk_env_layers.sh"

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

in_main_shell() {
  [[ "${BASHPID:-$$}" == "$ONSTART_MAIN_PID" ]]
}

print_failed_marker() {
  # Only the main shell prints markers, and only once.
  in_main_shell || return 0
  (( ONSTART_MARKER_PRINTED )) && return 0
  ONSTART_MARKER_PRINTED=1
  echo "========================================"
  echo "VAST_ONSTART FAILED: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "TOTAL ELAPSED: $(format_duration "$(elapsed_since_start)")"
  echo "LOG FILE: $ONSTART_LOG"
  echo "========================================"
}

die() {
  printf '[%s] [%s] ERROR: %s\n' "$SCRIPT_NAME" "$(date -u '+%H:%M:%S')" "$*" >&2
  trap - ERR
  print_failed_marker
  exit 1
}

report_unhandled_failure() {
  local status=$?
  local line="${BASH_LINENO[0]:-${LINENO:-unknown}}"
  if ! in_main_shell; then
    # A subshell failed: let the parent's own failure handling report it once.
    exit "$status"
  fi
  trap - ERR
  printf '[%s] [%s] ERROR: unhandled failure near line %s (exit %s)\n' \
    "$SCRIPT_NAME" "$(date -u '+%H:%M:%S')" "$line" "$status" >&2
  print_failed_marker
  exit "$status"
}

cleanup_bootstrap_secret() {
  in_main_shell || return 0
  if [[ -n "$BOOTSTRAP_SECRET_TMP" && -f "$BOOTSTRAP_SECRET_TMP" ]]; then
    rm -f -- "$BOOTSTRAP_SECRET_TMP"
  fi
}

report_exit() {
  # Catches exits the ERR trap cannot see (set -u unbound variables, explicit exit N).
  local status=$?
  cleanup_bootstrap_secret
  if (( status != 0 )) && in_main_shell && (( ! ONSTART_MARKER_PRINTED )); then
    printf '[%s] [%s] ERROR: exiting with status %s\n' "$SCRIPT_NAME" "$(date -u '+%H:%M:%S')" "$status" >&2
    print_failed_marker
  fi
}

report_signal() {
  local signal="$1"
  trap - ERR
  printf '[%s] [%s] ERROR: received SIG%s\n' "$SCRIPT_NAME" "$(date -u '+%H:%M:%S')" "$signal" >&2
  print_failed_marker
  case "$signal" in
    INT) exit 130 ;;
    HUP) exit 129 ;;
    *) exit 143 ;;
  esac
}

trap report_unhandled_failure ERR
trap report_exit EXIT
trap 'report_signal TERM' TERM
trap 'report_signal INT' INT
trap 'report_signal HUP' HUP

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

legacy_recipe_selected() {
  [[ "$ONSTART_RECIPE" == "legacy_int8" ]]
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

detect_public_ip() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 5 https://api.ipify.org || true
  fi
}

set_env_file_value() {
  local key="$1"
  local value="$2"
  local repaired_env
  repaired_env="$(mktemp "${TURN_ENV_FILE}.XXXXXX")"
  awk -v key="$key" -v value="$value" '
    BEGIN { found = 0 }
    index($0, key "=") == 1 {
      if (!found) print key "=" value
      found = 1
      next
    }
    { print }
    END { if (!found) print key "=" value }
  ' "$TURN_ENV_FILE" > "$repaired_env"
  chmod 600 "$repaired_env"
  mv "$repaired_env" "$TURN_ENV_FILE"
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
  if env_flag_is_true "$IMMUTABLE_BOOT"; then
    die "Immutable image is missing coturn; rebuild the image (runtime apt installation is forbidden)"
  fi

  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    die "WebRTC TURN autostart requires coturn, but turnserver is not installed and this script is not running as root"
  fi

  log "Installing coturn for WebRTC TURN autostart"
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get install -y coturn
}

installer_group_args() {
  # Groups must match between --check and the install, so both use this list.
  INSTALLER_GROUP_ARGS=(--venv "$VENV_PATH")
  if avatar_prep_requested; then
    INSTALLER_GROUP_ARGS+=(--with-avatar-prep)
  fi
  if legacy_recipe_selected; then
    INSTALLER_GROUP_ARGS+=(--with-legacy-int8)
  fi
  case "${SETUP_KOKORO:-}" in
    "") ;;
    1|true|yes|on|with) INSTALLER_GROUP_ARGS+=(--with-kokoro) ;;
    0|false|no|off|without) INSTALLER_GROUP_ARGS+=(--without-kokoro) ;;
    *) die "Unsupported SETUP_KOKORO value: $SETUP_KOKORO" ;;
  esac
  case "${SETUP_NATIVE_VP8:-}" in
    ""|auto) ;;
    1|true|yes|on|with) INSTALLER_GROUP_ARGS+=(--with-native-vp8) ;;
    0|false|no|off|without) INSTALLER_GROUP_ARGS+=(--without-native-vp8) ;;
    *) die "Unsupported SETUP_NATIVE_VP8 value: $SETUP_NATIVE_VP8" ;;
  esac
  if env_flag_is_true "${SETUP_CHIN_TOOLS:-0}"; then
    INSTALLER_GROUP_ARGS+=(--with-chin-tools)
  fi
  if [[ -n "${SETUP_MATRIX:-}" ]]; then
    INSTALLER_GROUP_ARGS+=(--matrix "$SETUP_MATRIX")
  fi
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    INSTALLER_GROUP_ARGS+=(--python "$PYTHON_BIN")
  fi
}

run_install_check() {
  # Sets INSTALL_CHECK_RC (0 ok, 10 needs clean install, 11 repairable, other = error).
  # SETUP_CHECK_IMPORTS=1 adds the installer's CPU import smoke (CUDA hidden, ~10-30 s).
  local check_args=(--check)
  if env_flag_is_true "${SETUP_CHECK_IMPORTS:-0}"; then
    check_args+=(--check-imports)
  fi
  INSTALL_CHECK_RC=0
  log "Running install check: scripts/install_musetalk.sh ${check_args[*]} ${INSTALLER_GROUP_ARGS[*]}"
  bash "$INSTALLER" "${check_args[@]}" "${INSTALLER_GROUP_ARGS[@]}" || INSTALL_CHECK_RC=$?
  case "$INSTALL_CHECK_RC" in
    0) log "Install check: OK" ;;
    10) log "Install check: venv missing or built for an incompatible matrix (exit 10) -> needs a clean install" ;;
    11) log "Install check: incomplete but repairable in place (exit 11)" ;;
    *) log "⚠️  Install check exited $INSTALL_CHECK_RC (unexpected)" ;;
  esac
}

run_setup_if_needed() {
  export HF_MAX_WORKERS
  installer_group_args

  if [[ -n "${ARTIFACT_DIR:-}" ]]; then
    log "ARTIFACT_DIR=$ARTIFACT_DIR is ignored (TensorRT engines are managed by scripts/unet_engine_store.py)"
  fi

  if [[ ! -f "$INSTALLER" ]]; then
    if env_flag_is_true "$IMMUTABLE_BOOT"; then
      die "Immutable image is missing its canonical installer/checker: $INSTALLER"
    fi
    if ! env_flag_is_true "$AUTO_SETUP"; then
      [[ -x "$VENV_PATH/bin/python" ]] || die "AUTO_SETUP=0 and the venv python is missing ($VENV_PATH/bin/python); installer $INSTALLER not found either"
      log "⚠️  Installer $INSTALLER not found; AUTO_SETUP=0 so continuing with the existing venv"
      return 0
    fi
    die "Installer not found: $INSTALLER"
  fi

  run_install_check

  if env_flag_is_true "$IMMUTABLE_BOOT" && (( INSTALL_CHECK_RC != 0 )); then
    die "Immutable image install check failed (exit $INSTALL_CHECK_RC); rebuild instead of repairing at boot"
  fi

  if ! env_flag_is_true "$AUTO_SETUP"; then
    log "AUTO_SETUP disabled: check only"
    [[ -x "$VENV_PATH/bin/python" ]] || die "AUTO_SETUP=0 but the venv python is missing: $VENV_PATH/bin/python"
    case "$INSTALL_CHECK_RC" in
      0) ;;
      10) die "AUTO_SETUP=0 but the install check says the venv needs a clean install (exit 10); rerun with AUTO_SETUP=1 (optionally SETUP_CLEAN=1)" ;;
      11) log "⚠️  Install is incomplete (exit 11) but AUTO_SETUP=0; continuing. Repair with: bash scripts/install_musetalk.sh ${INSTALLER_GROUP_ARGS[*]}" ;;
      *) log "⚠️  Install check exited $INSTALL_CHECK_RC but AUTO_SETUP=0; continuing" ;;
    esac
    return 0
  fi

  local install_args=("${INSTALLER_GROUP_ARGS[@]}")
  local mode="repair"
  if env_flag_is_true "$SETUP_CLEAN" || (( INSTALL_CHECK_RC == 10 )); then
    mode="clean"
    install_args+=(--clean)
  elif (( INSTALL_CHECK_RC == 0 )); then
    log "Existing install looks valid; skipping setup"
    return 0
  fi

  case "${SETUP_SKIP_APT,,}" in
    auto)
      if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
        install_args+=(--skip-apt)
      fi
      ;;
    1|true|yes|on)
      install_args+=(--skip-apt)
      ;;
    0|false|no|off)
      ;;
    *)
      die "Unsupported SETUP_SKIP_APT value: $SETUP_SKIP_APT"
      ;;
  esac
  if env_flag_is_true "$SETUP_SKIP_WEIGHTS"; then
    install_args+=(--skip-weights)
  fi
  if [[ -n "${SETUP_SELFTEST:-}" ]] && ! env_flag_is_true "$SETUP_SELFTEST"; then
    install_args+=(--no-selftest)
  fi

  if avatar_prep_requested; then
    log "Full-stack/avatar-prep install requested (mmcv/mmdet/mmpose + avatar-prep weights)"
  else
    log "Using the server-only install; enable SETUP_FULL_STACK=1 only if this node must handle /avatars/prepare"
  fi
  log "Using HF_MAX_WORKERS=$HF_MAX_WORKERS for setup/download flow"
  log "Running install ($mode): scripts/install_musetalk.sh ${install_args[*]}"
  bash "$INSTALLER" "${install_args[@]}"
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
  if ! legacy_recipe_selected; then
    model_files+=("models/taesd/config.json" "models/taesd/diffusion_pytorch_model.safetensors")
  fi

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

  # Check Python imports (ONSTART_POST_VALIDATE_IMPORTS=0 skips them; the install check
  # already ran an import smoke with CUDA hidden)
  if [[ -x "$PY" ]] && env_flag_is_true "$ONSTART_POST_VALIDATE_IMPORTS"; then
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
    if env_flag_is_true "$IMMUTABLE_BOOT"; then
      die "Immutable image post-setup validation failed"
    fi
  fi
}

immutable_runtime_policy() {
  env_flag_is_true "$IMMUTABLE_BOOT" || return 0
  local policy_file="$REPO_ROOT/docker/musetalk/release.py" policy_exports
  [[ -f "$policy_file" ]] || die "Immutable image release verifier missing"
  [[ -f "${MUSETALK_RELEASE_MANIFEST:-}" ]] || die "Immutable image release manifest missing"
  policy_exports="$("$VENV_PATH/bin/python" -B "$policy_file" policy --root "$REPO_ROOT" \
    --manifest "$MUSETALK_RELEASE_MANIFEST")" || die "Immutable image release policy invalid"
  # The checked-in verifier emits only validated fixed policy keys, never secret values.
  eval "$policy_exports"
}

bootstrap_runtime_secrets() {
  # Images need authorized private model inputs before the full install check.
  # Reuse this canonical bootstrap exactly once; source installs retain old order.
  if env_flag_is_true "$IMMUTABLE_BOOT" && (( IMMUTABLE_SECRETS_LOADED )); then
    log "Immutable runtime secrets already bootstrapped before model verification"
    return 0
  fi
  local secret_id="${MUSETALK_AWS_SECRET_ID:-}"
  if [[ -z "$secret_id" ]]; then
    if env_flag_is_true "$IMMUTABLE_BOOT"; then IMMUTABLE_SECRETS_LOADED=1; fi
    log "MuseTalk AWS Secrets Manager bootstrap skipped (MUSETALK_AWS_SECRET_ID not set)"
    return 0
  fi

  local PY="$VENV_PATH/bin/python"
  [[ -x "$PY" ]] || die "Cannot bootstrap runtime secrets; venv Python not found at $PY"

  local strict="${MUSETALK_SECRETS_STRICT:-${SECRETS_STRICT:-true}}"
  local verify_s3="${MUSETALK_SECRETS_VERIFY_S3:-1}"
  local tmp_env
  if env_flag_is_true "$IMMUTABLE_BOOT"; then
    [[ -d "${MUSETALK_BOOTSTRAP_SECRET_DIR:-}" ]] || die "Immutable bootstrap secret state directory missing"
    tmp_env="$(mktemp "$MUSETALK_BOOTSTRAP_SECRET_DIR/secret-XXXXXX.env")"
  else
    tmp_env="$(mktemp "$WORKSPACE_ROOT/.musetalk-runtime-secret.XXXXXX.env")"
  fi
  # Register cleanup before writing any credentials. A stale secret cannot
  # redirect cleanup to another path, and EXIT runs after source/TERM failure.
  readonly BOOTSTRAP_SECRET_TMP="$tmp_env"
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
    cleanup_bootstrap_secret
    if env_flag_is_true "$IMMUTABLE_BOOT"; then IMMUTABLE_SECRETS_LOADED=1; fi
    log "MuseTalk runtime secret env exports loaded"
    return 0
  fi

  cleanup_bootstrap_secret
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
    local detected_public_ip=""
    if ! env_flag_is_true "${TURN_PUBLIC_IP_PINNED:-0}"; then
      detected_public_ip="$(detect_public_ip)"
    fi
    if [[ -n "$detected_public_ip" && "$detected_public_ip" != "${TURN_PUBLIC_IP:-}" ]]; then
      set_env_file_value TURN_PUBLIC_IP "$detected_public_ip"
      log "Corrected stale TURN public IP ${TURN_PUBLIC_IP:-unset} -> $detected_public_ip"
      TURN_PUBLIC_IP="$detected_public_ip"
      unset WEBRTC_TURN_URLS WEBRTC_SERVER_TURN_URLS
    fi
    local existing_vast_tcp_1455 existing_vast_udp_3478
    existing_vast_tcp_1455="${VAST_TCP_PORT_1455:-$(read_proc1_env VAST_TCP_PORT_1455)}"
    existing_vast_udp_3478="${VAST_UDP_PORT_3478:-$(read_proc1_env VAST_UDP_PORT_3478)}"
    if env_flag_is_true "${TURN_PREFER_UDP:-1}" && [[ -n "$existing_vast_udp_3478" ]]; then
      if [[ "${TURN_PUBLIC_TRANSPORT:-}" != "udp" || "${TURN_PUBLIC_PORT:-}" != "$existing_vast_udp_3478" || "${TURN_LISTEN_PORT:-}" != "3478" ]]; then
        log "Promoting TURN media transport from ${TURN_PUBLIC_TRANSPORT:-unset} to UDP"
        set_env_file_value TURN_PUBLIC_TRANSPORT udp
        set_env_file_value TURN_PUBLIC_PORT "$existing_vast_udp_3478"
        set_env_file_value TURN_LISTEN_PORT 3478
        TURN_PUBLIC_TRANSPORT=udp
        TURN_PUBLIC_PORT="$existing_vast_udp_3478"
        TURN_LISTEN_PORT=3478
        unset WEBRTC_TURN_URLS WEBRTC_SERVER_TURN_URLS
      fi
      if [[ -n "$existing_vast_tcp_1455" ]]; then
        set_env_file_value TURN_TCP_FALLBACK_LISTEN_PORT 1455
        set_env_file_value TURN_TCP_FALLBACK_PUBLIC_PORT "$existing_vast_tcp_1455"
        TURN_TCP_FALLBACK_LISTEN_PORT=1455
        TURN_TCP_FALLBACK_PUBLIC_PORT="$existing_vast_tcp_1455"
      fi
    fi
    export TURN_ENV_FILE WEBRTC_RELAY_ENABLED WEBRTC_TURN_AUTOSTART
    if env_flag_is_true "${WEBRTC_TURN_AUTOSTART:-0}"; then
      ensure_coturn_available
    fi
    log "Loaded existing WebRTC TURN env from $TURN_ENV_FILE"
    return 0
  fi

  local public_ip detected_public_ip vast_tcp_1455 vast_udp_3478 listen_port public_port transport turn_pass tcp_fallback_listen_port tcp_fallback_public_port
  detected_public_ip="$(detect_public_ip)"
  public_ip="${TURN_PUBLIC_IP:-${PUBLIC_IP:-${detected_public_ip:-${PUBLIC_IPADDR:-$(read_proc1_env PUBLIC_IPADDR)}}}}"
  vast_tcp_1455="${VAST_TCP_PORT_1455:-$(read_proc1_env VAST_TCP_PORT_1455)}"
  vast_udp_3478="${VAST_UDP_PORT_3478:-$(read_proc1_env VAST_UDP_PORT_3478)}"

  if [[ -z "$public_ip" ]]; then
    if webrtc_turn_required; then
      die "SETUP_WEBRTC_TURN=$SETUP_WEBRTC_TURN but no TURN_PUBLIC_IP/PUBLIC_IPADDR could be detected"
    fi
    log "WebRTC TURN auto bootstrap skipped: no public IP detected"
    return 0
  fi

  if [[ -n "$vast_udp_3478" ]]; then
    listen_port="${TURN_LISTEN_PORT:-3478}"
    public_port="${TURN_PUBLIC_PORT:-$vast_udp_3478}"
    transport="${TURN_PUBLIC_TRANSPORT:-udp}"
  elif [[ -n "$vast_tcp_1455" ]]; then
    listen_port="${TURN_LISTEN_PORT:-1455}"
    public_port="${TURN_PUBLIC_PORT:-$vast_tcp_1455}"
    transport="${TURN_PUBLIC_TRANSPORT:-tcp}"
  else
    if webrtc_turn_required; then
      die "SETUP_WEBRTC_TURN=$SETUP_WEBRTC_TURN but no Vast TURN port mapping was detected (expected VAST_TCP_PORT_1455 or VAST_UDP_PORT_3478)"
    fi
    log "WebRTC TURN auto bootstrap skipped: no Vast TURN port mapping detected"
    return 0
  fi

  tcp_fallback_listen_port="${TURN_TCP_FALLBACK_LISTEN_PORT:-}"
  tcp_fallback_public_port="${TURN_TCP_FALLBACK_PUBLIC_PORT:-}"
  if [[ "$transport" == "udp" && -n "$vast_tcp_1455" ]]; then
    tcp_fallback_listen_port="${tcp_fallback_listen_port:-1455}"
    tcp_fallback_public_port="${tcp_fallback_public_port:-$vast_tcp_1455}"
  fi

  turn_pass="${TURN_PASS:-${WEBRTC_TURN_PASS:-$(generate_turn_password)}}"
  mkdir -p "$(dirname "$TURN_ENV_FILE")"
  umask 077
  cat > "$TURN_ENV_FILE" <<EOF
WEBRTC_RELAY_ENABLED=1
WEBRTC_TURN_AUTOSTART=1

TURN_PUBLIC_IP=$public_ip
TURN_PUBLIC_IP_PINNED=0
TURN_PUBLIC_PORT=$public_port
TURN_PUBLIC_TRANSPORT=$transport
TURN_LISTEN_PORT=$listen_port
TURN_TCP_FALLBACK_LISTEN_PORT=$tcp_fallback_listen_port
TURN_TCP_FALLBACK_PUBLIC_PORT=$tcp_fallback_public_port
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

select_best_trt_profile() {
  if ! env_flag_is_true "$MUSETALK_SELECT_BEST_TRT_PROFILE"; then
    log "TRT profile selection disabled (MUSETALK_SELECT_BEST_TRT_PROFILE=$MUSETALK_SELECT_BEST_TRT_PROFILE)"
    return 0
  fi

  local PY="$VENV_PATH/bin/python"
  if [[ ! -x "$PY" ]]; then
    die "Cannot select TRT profile; venv Python not found at $PY"
  fi
  if [[ ! -f "$REPO_ROOT/scripts/select_unet_trt_profile.py" ]]; then
    die "Cannot select TRT profile; selector script is missing"
  fi

  log "Selecting best validated TRT profile"
  if (
    cd "$REPO_ROOT"
    "$PY" "$REPO_ROOT/scripts/select_unet_trt_profile.py" \
      --output "$MUSETALK_TRT_PROFILE_ENV_FILE" \
      --prefer "$MUSETALK_TRT_PROFILE_PREFER"
  ); then
    if [[ -f "$MUSETALK_TRT_PROFILE_ENV_FILE" ]]; then
      log "TRT profile env ready at $MUSETALK_TRT_PROFILE_ENV_FILE"
    else
      die "TRT selector completed without creating $MUSETALK_TRT_PROFILE_ENV_FILE"
    fi
    return 0
  fi

  die "TRT profile selection failed; restored artifacts are missing or invalid"
}

restore_trt_artifacts() {
  local restore_mode="${MUSETALK_TRT_ARTIFACT_RESTORE:-auto}"
  case "${restore_mode,,}" in
    0|false|no|off)
      log "TRT artifact restore disabled (MUSETALK_TRT_ARTIFACT_RESTORE=$restore_mode)"
      return 0
      ;;
  esac

  local uri="${MUSETALK_TRT_ARTIFACT_URI:-}"
  if [[ -z "$uri" && -n "${TRT_ARTIFACT_S3_BUCKET:-}" ]]; then
    uri="s3://${TRT_ARTIFACT_S3_BUCKET}/${MUSETALK_TRT_ARTIFACT_KEY}"
  fi
  if [[ -z "$uri" ]]; then
    if [[ "${restore_mode,,}" == "auto" ]]; then
      log "TRT artifact restore skipped; set MUSETALK_TRT_ARTIFACT_URI or TRT_ARTIFACT_S3_BUCKET"
      return 0
    fi
    die "TRT artifact restore requested, but MUSETALK_TRT_ARTIFACT_URI/TRT_ARTIFACT_S3_BUCKET is not set"
  fi

  local PY="$VENV_PATH/bin/python"
  [[ -x "$PY" ]] || die "Cannot restore TRT artifacts; venv Python not found at $PY"
  [[ -f "$REPO_ROOT/scripts/trt_artifact_bundle.py" ]] || die "TRT artifact restore script missing"

  log "Restoring TRT artifact bundle from $uri"
  local args=(--repo-root "$REPO_ROOT")
  if env_flag_is_true "$MUSETALK_TRT_ARTIFACT_STRICT"; then
    args+=(--strict)
  fi
  if [[ -n "$MUSETALK_TRT_ARTIFACT_SHA256" ]]; then
    args+=(restore --uri "$uri" --expected-sha256 "$MUSETALK_TRT_ARTIFACT_SHA256")
  else
    args+=(restore --uri "$uri")
  fi
  if (
    cd "$REPO_ROOT"
    "$PY" "$REPO_ROOT/scripts/trt_artifact_bundle.py" "${args[@]}"
  ); then
    log "TRT artifact restore complete"
    return 0
  fi

  die "TRT artifact restore failed from $uri"
}

refresh_recipe_after_secrets() {
  # The runtime secret may export MUSETALK_RECIPE (one central switch for every worker). The install
  # check already ran with the recipe known at boot, so only a switch among the fast-family recipes
  # (same install groups) is taken over; a switch to or from legacy_int8 must be set in the template.
  mt_env_effective_recipe "$REPO_ROOT"
  [[ "$MT_ENV_RECIPE" == "$ONSTART_RECIPE" ]] && return 0
  case "$MT_ENV_RECIPE" in
    fast|fast300|r5) ;;
    *) die "MUSETALK_RECIPE=$MT_ENV_RECIPE (source=$MT_ENV_RECIPE_SOURCE, after the secret bootstrap) is not fast, fast300 or r5; set legacy_int8 in the template" ;;
  esac
  if legacy_recipe_selected; then
    die "MUSETALK_RECIPE changed from legacy_int8 to $MT_ENV_RECIPE after the secret bootstrap; set it in the template instead"
  fi
  log "recipe=$MT_ENV_RECIPE (source=$MT_ENV_RECIPE_SOURCE, after the secret bootstrap; was $ONSTART_RECIPE)"
  ONSTART_RECIPE="$MT_ENV_RECIPE"
  ONSTART_RECIPE_SOURCE="$MT_ENV_RECIPE_SOURCE"
}

restore_r5_bundle() {
  # Sets R5_BUNDLE_ACTIVE=1 when a bundle's engines are on disk, verified and stamped for the resolver.
  # Candidates come from the bundle:<a>|<b> prerequisite of configs/recipes/r5.env, in order of preference
  # (a GPU-specific bundle first, the portable AMPERE_PLUS one last); the resolver applies the same host rule
  # (bundle-check) and the same stamp, so it serves exactly what is restored here.
  local mode="${MUSETALK_R5_BUNDLE_RESTORE:-required}" recipe_file="$REPO_ROOT/configs/recipes/r5.env"
  local PY="$VENV_PATH/bin/python" candidates report="" rc=0 name state why uri descriptor tried=0
  local -a fits=() fields=()
  R5_BUNDLE_ACTIVE=0
  if env_flag_is_true "$IMMUTABLE_BOOT"; then
    [[ "$mode" == baked ]] || die "Immutable image requires baked native artifacts"
    "$PY" -B "$REPO_ROOT/docker/musetalk/release.py" runtime --root "$REPO_ROOT" \
      --venv "$VENV_PATH" --manifest "$MUSETALK_RELEASE_MANIFEST" \
      || die "Baked native release integrity/host verification failed"
    R5_BUNDLE_ACTIVE=1
    log "Baked native r5 artifacts verified on this host; no engine download/build"
    return 0
  fi
  case "${mode,,}" in
    0|false|no|off)
      log "r5 engine bundle restore disabled (MUSETALK_R5_BUNDLE_RESTORE=$mode); the resolver serves what is on disk"
      return 0
      ;;
    auto|required) ;;
    *) die "MUSETALK_R5_BUNDLE_RESTORE=$mode must be required, auto or off" ;;
  esac
  candidates="$(sed -nE 's/^# @lever r5_engines requires=(.*,)?bundle:([A-Za-z0-9._|-]+).*/\2/p' "$recipe_file" 2>/dev/null | head -n 1)"
  [[ -n "$candidates" ]] || die "$recipe_file names no bundle:<name> prerequisite for group r5_engines"
  [[ -x "$PY" ]] || die "Cannot restore the r5 engine bundle; venv Python not found at $PY"

  report="$("$PY" -B "$REPO_ROOT/scripts/musetalk_host_profile.py" bundle-check --bundle "$candidates" \
    --host-only --repo-root "$REPO_ROOT" --venv "$VENV_PATH" 2>/dev/null)" || rc=$?
  while IFS=$'\t' read -r name state why; do
    [[ -n "$name" ]] || continue
    if [[ "$state" == ok ]]; then
      fits+=("$name")
      log "r5 bundle candidate $name fits this host: $why"
    else
      log "r5 bundle candidate $name does not fit this host: $why"
    fi
  done <<< "$report"
  if (( ${#fits[@]} == 0 )); then
    log "⚠️  No r5 engine bundle fits this host (bundle-check exit $rc)."
    log "    The resolver drops the r5 engine group: eager UNet + compiled TAESD + r5 serving levers"
    return 0
  fi

  mkdir -p "$MUSETALK_TRT_ARTIFACT_STAGE_DIR"
  for name in "${fits[@]}"; do
    descriptor="$REPO_ROOT/configs/trt_bundles/$name.json"
    mapfile -t fields < <("$PY" -B -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("sha256", "")); print(d.get("s3_key", "")); print(d.get("sidecar_dir", "")); print(d.get("size_bytes", ""))' \
      "$descriptor" 2>/dev/null)
    local sha="${fields[0]:-}" s3_key="${fields[1]:-}" sidecar="${fields[2]:-}" size="${fields[3]:-}"
    [[ -n "$sha" && -n "$s3_key" && -n "$sidecar" ]] || die "$descriptor lacks sha256/s3_key/sidecar_dir"
    # MUSETALK_R5_BUNDLE_URI replaces the first candidate's URI only (tests, mirrors); the sha256 stays pinned.
    uri=""
    if (( tried == 0 )) && [[ -n "${MUSETALK_R5_BUNDLE_URI:-}" ]]; then
      uri="$MUSETALK_R5_BUNDLE_URI"
    elif [[ -n "${TRT_ARTIFACT_S3_BUCKET:-}" ]]; then
      uri="s3://${TRT_ARTIFACT_S3_BUCKET}/${s3_key}"
    fi
    tried=$(( tried + 1 ))
    if [[ -z "$uri" ]]; then
      log "⚠️  r5 engine bundle $name: no URI (set TRT_ARTIFACT_S3_BUCKET via the runtime secret, or MUSETALK_R5_BUNDLE_URI)"
      continue
    fi
    log "r5 engine bundle $name: $uri (${size:-?} bytes, sha256 ${sha:0:12}); staging in $MUSETALK_TRT_ARTIFACT_STAGE_DIR"
    # --skip-if-verified: a start whose stamp still matches and whose files still hash clean skips the download
    # (only when the checkout survives; the standard Vast template re-clones it on every start). The sidecars go
    # to $sidecar, never the repo root (that pair belongs to legacy_int8).
    if (
      cd "$REPO_ROOT"
      "$PY" -B "$REPO_ROOT/scripts/trt_artifact_bundle.py" --repo-root "$REPO_ROOT" --strict \
        --sidecar-dir "$sidecar" restore --uri "$uri" --expected-sha256 "$sha" \
        --stage-dir "$MUSETALK_TRT_ARTIFACT_STAGE_DIR" --skip-if-verified
    ); then
      R5_BUNDLE_ACTIVE=1
      log "✅ r5 engine bundle $name ready (stamp: $sidecar/.musetalk_trt_artifact_restored.json)"
      return 0
    fi
    log "⚠️  r5 engine bundle $name could not be restored from $uri; trying the next candidate"
  done
  if [[ "${mode,,}" == "auto" ]]; then
    log "⚠️  No r5 engine bundle restored (MUSETALK_R5_BUNDLE_RESTORE=auto): serving without it"
    return 0
  fi
  if [[ -z "${TRT_ARTIFACT_S3_BUCKET:-}${MUSETALK_R5_BUNDLE_URI:-}" ]]; then
    die "r5 engine bundle required on this host (${fits[*]}), but TRT_ARTIFACT_S3_BUCKET/MUSETALK_R5_BUNDLE_URI is not set"
  fi
  die "No r5 engine bundle could be restored (${fits[*]}); set MUSETALK_R5_BUNDLE_RESTORE=auto to boot without it"
}

# Engine-store knobs an operator may keep in an overrides file. unet_engine_store.py reads only its
# own environment while the resolver also reads the overrides files, so forward them (the caller
# env still wins) to keep `ensure` and the resolver on the same store roots / remotes.
ENGINE_STORE_ENV_KEYS=(
  MUSETALK_UNET_ENGINE_STORE MUSETALK_UNET_STAGEWISE_ENGINE_STORE MUSETALK_TAESD_TRT_ENGINE_STORE
  MUSETALK_UNET_ENGINE_REMOTE MUSETALK_UNET_STAGEWISE_ENGINE_REMOTE MUSETALK_TAESD_TRT_ENGINE_REMOTE
  MUSETALK_ENGINE_REMOTE_BASE MUSETALK_UNET_ENGINE_PUBLISH MUSETALK_ENGINE_AUTO_BUILD MUSETALK_UNET_ADOPT_PATHS
  MUSETALK_UNET_VALIDATION_CORPUS MUSETALK_ENGINE_LOG_DIR MUSETALK_UNET_BUILD_MIN_MEM_AVAILABLE_GB
  MUSETALK_TAESD_TRT_BATCH
)

engine_store_env_args() {
  # Sets ENGINE_STORE_ENV_ARGS to KEY=VALUE words for `env` (overrides-file values of unset keys).
  ENGINE_STORE_ENV_ARGS=()
  local key
  for key in "${ENGINE_STORE_ENV_KEYS[@]}"; do
    [[ -n "${!key+x}" ]] && continue
    if mt_env_peek "$key" "$REPO_ROOT"; then
      ENGINE_STORE_ENV_ARGS+=("$key=$MT_ENV_PEEK_VALUE")
    fi
  done
}

provision_engine_kind() {
  # provision_engine_kind KIND MODE REQUIRED [--batch N]
  local kind="$1" mode="$2" required="$3"
  shift 3
  local PY="$VENV_PATH/bin/python" rc=0 started
  # cwd is the repo root and the store defaults its repo root to its own checkout.
  local args=(ensure --kind "$kind" "$@" --provision "$mode")
  if (( required )); then
    args+=(--require)
  fi
  if [[ "${mode,,}" == "off" ]]; then
    log "Engine provisioning for $kind disabled (provision=off)"
    return 0
  fi
  started="$(date +%s)"
  engine_store_env_args
  log "Engine provisioning: $kind (provision=$mode required=$required${*:+ $*})"
  if (( ${#ENGINE_STORE_ENV_ARGS[@]} > 0 )); then
    log "Engine store settings from the overrides files: ${ENGINE_STORE_ENV_ARGS[*]%%=*}"
  fi
  (
    cd "$REPO_ROOT"
    env "${ENGINE_STORE_ENV_ARGS[@]}" "$PY" "$ENGINE_STORE" "${args[@]}"
  ) || rc=$?
  local took="$(( $(date +%s) - started ))"
  case "$rc" in
    0)
      log "✅ Engine $kind usable ($(format_duration "$took"))"
      ;;
    2|3)
      # ensure: 3 = no usable engine; with --require it reports that as 2.
      if (( required )); then
        die "No usable $kind engine and MUSETALK_UNET_MODE=trt requires one (exit $rc, $(format_duration "$took"))"
      fi
      if (( rc == 2 )); then
        log "⚠️  Engine provisioning for $kind exited 2 ($(format_duration "$took")); continuing (non-fatal)"
        return 0
      fi
      log "⚠️  No usable $kind engine ($(format_duration "$took")); the resolver will pick the fallback backend"
      ;;
    *)
      if (( required )); then
        die "Engine provisioning for $kind failed with exit $rc ($(format_duration "$took"))"
      fi
      log "⚠️  Engine provisioning for $kind exited $rc ($(format_duration "$took")); continuing (non-fatal)"
      ;;
  esac
}

provision_engines() {
  local unet_mode unet_required=0 stagewise_batch unet_ts_default=auto
  if [[ ! -f "$ENGINE_STORE" ]]; then
    log "⚠️  Engine store $ENGINE_STORE not found; skipping engine provisioning"
    return 0
  fi
  [[ -x "$VENV_PATH/bin/python" ]] || die "Cannot provision engines; venv Python not found at $VENV_PATH/bin/python"
  unet_mode="$(mt_env_peek_or MUSETALK_UNET_MODE "$REPO_ROOT" auto)"
  if [[ "${unet_mode,,}" == "trt" ]]; then
    unet_required=1
  fi
  if [[ "$ONSTART_RECIPE" == "r5" ]] && (( !unet_required )); then
    # r5 serves the bundle's stagewise UNet; the old .ts engine (a multi-minute build) is not built by default.
    unet_ts_default=off
    if (( R5_BUNDLE_ACTIVE )); then
      log "Recipe r5: the bundle's engines serve this host; .ts UNet provisioning off (MUSETALK_UNET_ENGINE_PROVISION=auto builds it)"
    else
      log "⚠️  Recipe r5 without its bundle on this host: the UNet runs eager; .ts provisioning stays off unless MUSETALK_UNET_ENGINE_PROVISION=auto"
    fi
  fi
  provision_engine_kind unet_ts \
    "$(mt_env_peek_or MUSETALK_UNET_ENGINE_PROVISION "$REPO_ROOT" "$unet_ts_default")" "$unet_required"
  if [[ "$ONSTART_RECIPE" == "fast300" ]]; then
    provision_engine_kind taesd_trt \
      "$(mt_env_peek_or MUSETALK_TAESD_TRT_PROVISION "$REPO_ROOT" auto)" 0
    stagewise_batch="$(mt_env_peek_or MUSETALK_UNET_STAGEWISE_BATCH "$REPO_ROOT" 16)"
    local batch_args=()
    if [[ "$stagewise_batch" != "16" ]]; then
      batch_args=(--batch "$stagewise_batch")
    fi
    provision_engine_kind unet_stagewise \
      "$(mt_env_peek_or MUSETALK_UNET_STAGEWISE_PROVISION "$REPO_ROOT" auto)" 0 "${batch_args[@]}"
  fi
}

main() {
  local phase_start phase_elapsed total_elapsed
  immutable_runtime_policy
  local setup_mode="server-only"
  if full_stack_requested; then
    setup_mode="full-stack"
  elif avatar_prep_requested; then
    setup_mode="server+avatar-prep"
  fi

  mt_env_effective_recipe "$REPO_ROOT"
  ONSTART_RECIPE="$MT_ENV_RECIPE"
  ONSTART_RECIPE_SOURCE="$MT_ENV_RECIPE_SOURCE"
  case "$ONSTART_RECIPE" in
    fast|fast300|r5|legacy_int8) ;;
    *) die "Unsupported MUSETALK_RECIPE=$ONSTART_RECIPE (source=$ONSTART_RECIPE_SOURCE); expected fast, fast300, r5 or legacy_int8" ;;
  esac

  log "Vast.ai MuseTalk on-start begin"
  log "recipe=$ONSTART_RECIPE (source=$ONSTART_RECIPE_SOURCE)"
  log "repo=$REPO_ROOT"
  log "workspace=$WORKSPACE_ROOT"
  log "venv=$VENV_PATH"
  log "profile=$PROFILE host=$HOST port=$PORT"
  log "setup_mode=$setup_mode"
  log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
  log "Disk free: $(df -h /workspace | tail -1 | awk '{print $4}')"

  if env_flag_is_true "$IMMUTABLE_BOOT"; then
    phase_start="$(date +%s)"
    log "Immutable runtime secret bootstrap/private model verification begin"
    bootstrap_runtime_secrets
    immutable_runtime_policy
    "$VENV_PATH/bin/python" -B "$REPO_ROOT/docker/musetalk/release.py" runtime-models \
      --root "$REPO_ROOT" --manifest "$MUSETALK_RELEASE_MANIFEST" \
      --cache "$WORKSPACE_ROOT/private-model-cache"
    phase_elapsed="$(( $(date +%s) - phase_start ))"
    log "Immutable private model restore phase finished in $(format_duration "$phase_elapsed")"
  fi

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
  immutable_runtime_policy
  refresh_recipe_after_secrets
  phase_elapsed="$(( $(date +%s) - phase_start ))"
  log "Runtime secret bootstrap phase finished in $(format_duration "$phase_elapsed")"

  phase_start="$(date +%s)"
  configure_webrtc_turn
  phase_elapsed="$(( $(date +%s) - phase_start ))"
  log "WebRTC TURN bootstrap phase finished in $(format_duration "$phase_elapsed")"

  if legacy_recipe_selected; then
    phase_start="$(date +%s)"
    restore_trt_artifacts
    phase_elapsed="$(( $(date +%s) - phase_start ))"
    log "TRT artifact restore phase finished in $(format_duration "$phase_elapsed")"

    phase_start="$(date +%s)"
    select_best_trt_profile
    phase_elapsed="$(( $(date +%s) - phase_start ))"
    log "TRT profile selection phase finished in $(format_duration "$phase_elapsed")"

    phase_start="$(date +%s)"
    MUSETALK_RECIPE=legacy_int8 \
    PROFILE="$PROFILE" \
    HOST="$HOST" \
    PORT="$PORT" \
    REPO_ROOT="$REPO_ROOT" \
    VENV_PATH="$VENV_PATH" \
    MUSETALK_TRT_PROFILE_ENV_FILE="$MUSETALK_TRT_PROFILE_ENV_FILE" \
    MUSETALK_TRT_PROFILE_ENV_LOAD="${MUSETALK_TRT_PROFILE_ENV_LOAD:-1}" \
    bash "$REPO_ROOT/scripts/vast_server_ctl.sh" start
    phase_elapsed="$(( $(date +%s) - phase_start ))"
    log "Server start-to-health phase finished in $(format_duration "$phase_elapsed")"
  else
    log "Recipe $ONSTART_RECIPE: legacy TRT artifact restore and profile selection are skipped"
    if [[ "$ONSTART_RECIPE" == "r5" ]]; then
      phase_start="$(date +%s)"
      restore_r5_bundle
      phase_elapsed="$(( $(date +%s) - phase_start ))"
      log "r5 engine bundle phase finished in $(format_duration "$phase_elapsed")"
    fi
    phase_start="$(date +%s)"
    provision_engines
    phase_elapsed="$(( $(date +%s) - phase_start ))"
    log "Engine provisioning phase finished in $(format_duration "$phase_elapsed")"

    phase_start="$(date +%s)"
    HOST="$HOST" \
    PORT="$PORT" \
    REPO_ROOT="$REPO_ROOT" \
    VENV_PATH="$VENV_PATH" \
    bash "$REPO_ROOT/scripts/vast_server_ctl.sh" start
    phase_elapsed="$(( $(date +%s) - phase_start ))"
    log "Server start-to-health-and-verify phase finished in $(format_duration "$phase_elapsed")"
  fi

  total_elapsed="$(elapsed_since_start)"
  log "Overall on-start completed in $(format_duration "$total_elapsed")"
  log "Vast.ai MuseTalk on-start complete"

  ONSTART_MARKER_PRINTED=1
  echo "========================================"
  echo "VAST_ONSTART COMPLETE: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "TOTAL ELAPSED: $(format_duration "$total_elapsed")"
  echo "LOG FILE: $ONSTART_LOG"
  echo "========================================"
}

main "$@"
