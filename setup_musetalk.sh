#!/usr/bin/env bash
# Compatibility shim: the canonical installer is scripts/install_musetalk.sh (docs/STARTUP.md).
# Legacy flags from setup_trt_stagewise_server_env.sh are translated; anything else is passed
# through unchanged, so new installer flags work here too.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="${MUSETALK_INSTALLER:-$SCRIPT_DIR/scripts/install_musetalk.sh}"

log() {
  printf '[setup_musetalk.sh] %s\n' "$*"
}

die() {
  printf '[setup_musetalk.sh] ERROR: %s\n' "$*" >&2
  exit 1
}

args=()
with_avatar_prep=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-path)
      [[ $# -ge 2 ]] || die "--venv-path requires a value"
      args+=(--venv "$2")
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || die "--python-bin requires a value"
      args+=(--python "$2")
      shift 2
      ;;
    --artifact-dir)
      [[ $# -ge 2 ]] || die "--artifact-dir requires a value"
      log "ignoring --artifact-dir $2 (TensorRT engines are managed by scripts/unet_engine_store.py)"
      shift 2
      ;;
    --clean|--skip-apt|--skip-weights)
      args+=("$1")
      shift
      ;;
    --full-stack|--install-avatar-prep-deps)
      # Full stack = server runtime + avatar-prep (mmcv/mmdet/mmpose) deps and weights.
      with_avatar_prep=1
      shift
      ;;
    --install-modelopt)
      # nvidia-modelopt is only needed by the legacy INT8 SD-VAE recipe.
      args+=(--with-legacy-int8)
      shift
      ;;
    --skip-modelopt)
      log "--skip-modelopt is the default now (legacy INT8 deps are opt-in: --with-legacy-int8)"
      shift
      ;;
    --help|-h)
      cat <<EOF
Usage: setup_musetalk.sh [legacy flags] [install_musetalk.sh flags]

Shim for scripts/install_musetalk.sh. Legacy flags are translated:
  --venv-path P -> --venv P            --python-bin P -> --python P
  --full-stack / --install-avatar-prep-deps -> --with-avatar-prep
  --install-modelopt -> --with-legacy-int8   --skip-modelopt, --artifact-dir: ignored
  --clean, --skip-apt, --skip-weights: unchanged
Installer help follows.

EOF
      if [[ -f "$INSTALLER" ]]; then
        exec bash "$INSTALLER" --help
      fi
      exit 0
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done
if (( with_avatar_prep )); then
  args+=(--with-avatar-prep)
fi

[[ -f "$INSTALLER" ]] || die "installer not found: $INSTALLER"
log "delegating to scripts/install_musetalk.sh ${args[*]}"
exec bash "$INSTALLER" "${args[@]}"
