#!/usr/bin/env bash
# Launch the MuseTalk API with WebRTC forced through TURN relay.
#
# Final exec: ${MUSETALK_SERVER_LAUNCHER:-scripts/run_musetalk_server.sh} (which dispatches
# MUSETALK_RECIPE=legacy_int8 to the old scripts/run_trt_stagewise_server.sh). For the fast
# recipes this wrapper only sets the TURN/ICE knobs; the WebRTC sync/prebuffer defaults come
# from the overrides/resolved layers so an overrides file can still change them.

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_FILE="${TURN_ENV_FILE:-$REPO_ROOT/.env.webrtc-turn.local}"
SERVER_LAUNCHER="${MUSETALK_SERVER_LAUNCHER:-$REPO_ROOT/scripts/run_musetalk_server.sh}"
MT_ENV_LOG_PREFIX="$SCRIPT_NAME"
# shellcheck source=lib/musetalk_env_layers.sh
source "$SCRIPT_DIR/lib/musetalk_env_layers.sh"

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
}

die() {
  printf '[%s] ERROR: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

env_enabled() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
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

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

VAST_TCP_PORT_1455="${VAST_TCP_PORT_1455:-$(read_proc1_env VAST_TCP_PORT_1455)}"
VAST_UDP_PORT_3478="${VAST_UDP_PORT_3478:-$(read_proc1_env VAST_UDP_PORT_3478)}"
PUBLIC_IPADDR="${PUBLIC_IPADDR:-$(read_proc1_env PUBLIC_IPADDR)}"

if [[ -z "${TURN_LISTEN_PORT:-}" ]]; then
  if [[ -n "$VAST_UDP_PORT_3478" ]]; then
    TURN_LISTEN_PORT=3478
  elif [[ -n "$VAST_TCP_PORT_1455" ]]; then
    TURN_LISTEN_PORT=1455
  fi
fi
TURN_LISTEN_PORT="${TURN_LISTEN_PORT:-3478}"

if [[ -z "${TURN_PUBLIC_TRANSPORT:-}" ]]; then
  if [[ "$TURN_LISTEN_PORT" == "1455" && -n "$VAST_TCP_PORT_1455" ]]; then
    TURN_PUBLIC_TRANSPORT=tcp
  elif [[ "$TURN_LISTEN_PORT" == "3478" && -n "$VAST_UDP_PORT_3478" ]]; then
    TURN_PUBLIC_TRANSPORT=udp
  else
    TURN_PUBLIC_TRANSPORT=tcp
  fi
fi

if [[ -z "${TURN_PUBLIC_PORT:-}" ]]; then
  if [[ "$TURN_PUBLIC_TRANSPORT" == "tcp" && "$TURN_LISTEN_PORT" == "1455" && -n "$VAST_TCP_PORT_1455" ]]; then
    TURN_PUBLIC_PORT="$VAST_TCP_PORT_1455"
  elif [[ "$TURN_PUBLIC_TRANSPORT" == "udp" && "$TURN_LISTEN_PORT" == "3478" && -n "$VAST_UDP_PORT_3478" ]]; then
    TURN_PUBLIC_PORT="$VAST_UDP_PORT_3478"
  else
    TURN_PUBLIC_PORT="$TURN_LISTEN_PORT"
  fi
fi

TURN_TCP_FALLBACK_LISTEN_PORT="${TURN_TCP_FALLBACK_LISTEN_PORT:-}"
TURN_TCP_FALLBACK_PUBLIC_PORT="${TURN_TCP_FALLBACK_PUBLIC_PORT:-}"
if [[ "$TURN_PUBLIC_TRANSPORT" == "udp" && -n "$VAST_TCP_PORT_1455" ]]; then
  TURN_TCP_FALLBACK_LISTEN_PORT="${TURN_TCP_FALLBACK_LISTEN_PORT:-1455}"
  TURN_TCP_FALLBACK_PUBLIC_PORT="${TURN_TCP_FALLBACK_PUBLIC_PORT:-$VAST_TCP_PORT_1455}"
fi

TURN_PUBLIC_IP="${TURN_PUBLIC_IP:-${PUBLIC_IP:-${PUBLIC_IPADDR:-}}}"
TURN_USER="${TURN_USER:-${WEBRTC_TURN_USER:-webrtc}}"
TURN_PASS="${TURN_PASS:-${WEBRTC_TURN_PASS:-}}"
WEBRTC_USE_LOCAL_TURN="${WEBRTC_USE_LOCAL_TURN:-1}"

if [[ -z "${WEBRTC_TURN_URLS:-}" ]]; then
  if [[ -z "$TURN_PUBLIC_IP" ]]; then
    die "TURN_PUBLIC_IP is required to build WEBRTC_TURN_URLS. Set it in $ENV_FILE, or provide WEBRTC_TURN_URLS explicitly."
  fi
  WEBRTC_TURN_URLS="turn:$TURN_PUBLIC_IP:$TURN_PUBLIC_PORT?transport=$TURN_PUBLIC_TRANSPORT"
  if [[ -n "$TURN_TCP_FALLBACK_PUBLIC_PORT" ]]; then
    WEBRTC_TURN_URLS+=",turn:$TURN_PUBLIC_IP:$TURN_TCP_FALLBACK_PUBLIC_PORT?transport=tcp"
  fi
  export WEBRTC_TURN_URLS
fi

if [[ -z "$TURN_PASS" ]]; then
  die "TURN_PASS or WEBRTC_TURN_PASS is required. Set it in $ENV_FILE or the environment."
fi

export WEBRTC_ICE_TRANSPORT_POLICY="${WEBRTC_ICE_TRANSPORT_POLICY:-relay}"
export WEBRTC_STUN_URLS="${WEBRTC_STUN_URLS:-}"
if [[ -z "${WEBRTC_SERVER_TURN_URLS:-}" ]]; then
  if env_enabled "$WEBRTC_USE_LOCAL_TURN"; then
    if [[ -n "$TURN_TCP_FALLBACK_LISTEN_PORT" ]]; then
      export WEBRTC_SERVER_TURN_URLS="turn:127.0.0.1:$TURN_TCP_FALLBACK_LISTEN_PORT?transport=tcp"
    else
      export WEBRTC_SERVER_TURN_URLS="turn:127.0.0.1:$TURN_LISTEN_PORT?transport=$TURN_PUBLIC_TRANSPORT"
    fi
  else
    export WEBRTC_SERVER_TURN_URLS="$WEBRTC_TURN_URLS"
  fi
fi
export WEBRTC_TURN_USER="${WEBRTC_TURN_USER:-$TURN_USER}"
export WEBRTC_TURN_PASS="${WEBRTC_TURN_PASS:-$TURN_PASS}"
mt_env_effective_recipe "$REPO_ROOT"
if [[ "$MT_ENV_RECIPE" == "legacy_int8" ]]; then
  # Unchanged legacy behaviour.
  export WEBRTC_SYNC_MODE="${WEBRTC_SYNC_MODE:-strict_fifo}"
  export WEBRTC_VIDEO_PREBUFFER_SECONDS="${WEBRTC_VIDEO_PREBUFFER_SECONDS:-2.0}"
  export WEBRTC_AUDIO_PREBUFFER_SECONDS="${WEBRTC_AUDIO_PREBUFFER_SECONDS:-0.0}"
  export WEBRTC_ADAPTIVE_FPS="${WEBRTC_ADAPTIVE_FPS:-0}"
fi
# Fast recipes: the resolver emits WEBRTC_SYNC_MODE=strict_fifo, WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0
# and WEBRTC_ADAPTIVE_FPS=0 (same values); exporting them here would make them outrank the
# overrides files. WEBRTC_AUDIO_PREBUFFER_SECONDS is read by no Python module.

log "Starting API with WebRTC relay policy"
log "TURN env file=$ENV_FILE"
log "WEBRTC_ICE_TRANSPORT_POLICY=$WEBRTC_ICE_TRANSPORT_POLICY"
log "WEBRTC_TURN_URLS=$WEBRTC_TURN_URLS"
log "WEBRTC_SERVER_TURN_URLS=$WEBRTC_SERVER_TURN_URLS"
log "WEBRTC_USE_LOCAL_TURN=$WEBRTC_USE_LOCAL_TURN"
log "recipe=$MT_ENV_RECIPE (source=$MT_ENV_RECIPE_SOURCE) launcher=$SERVER_LAUNCHER"
log "WEBRTC_SYNC_MODE=${WEBRTC_SYNC_MODE:-<overrides/resolved layer>}"
log "WEBRTC_VIDEO_PREBUFFER_SECONDS=${WEBRTC_VIDEO_PREBUFFER_SECONDS:-<overrides/resolved layer>}"
log "WEBRTC_ADAPTIVE_FPS=${WEBRTC_ADAPTIVE_FPS:-<overrides/resolved layer>}"

exec bash "$SERVER_LAUNCHER" "$@"
