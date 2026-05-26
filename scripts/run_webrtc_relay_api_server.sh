#!/usr/bin/env bash
# Launch the current TRT-stagewise API with WebRTC forced through TURN relay.

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_FILE="${TURN_ENV_FILE:-$REPO_ROOT/.env.webrtc-turn.local}"

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

if [[ -z "${TURN_LISTEN_PORT:-}" && -n "$VAST_TCP_PORT_1455" ]]; then
  TURN_LISTEN_PORT=1455
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

TURN_PUBLIC_IP="${TURN_PUBLIC_IP:-${PUBLIC_IP:-${PUBLIC_IPADDR:-}}}"
TURN_USER="${TURN_USER:-${WEBRTC_TURN_USER:-webrtc}}"
TURN_PASS="${TURN_PASS:-${WEBRTC_TURN_PASS:-}}"
WEBRTC_USE_LOCAL_TURN="${WEBRTC_USE_LOCAL_TURN:-1}"

if [[ -z "${WEBRTC_TURN_URLS:-}" ]]; then
  if [[ -z "$TURN_PUBLIC_IP" ]]; then
    die "TURN_PUBLIC_IP is required to build WEBRTC_TURN_URLS. Set it in $ENV_FILE, or provide WEBRTC_TURN_URLS explicitly."
  fi
  export WEBRTC_TURN_URLS="turn:$TURN_PUBLIC_IP:$TURN_PUBLIC_PORT?transport=$TURN_PUBLIC_TRANSPORT"
fi

if [[ -z "$TURN_PASS" ]]; then
  die "TURN_PASS or WEBRTC_TURN_PASS is required. Set it in $ENV_FILE or the environment."
fi

export WEBRTC_ICE_TRANSPORT_POLICY="${WEBRTC_ICE_TRANSPORT_POLICY:-relay}"
export WEBRTC_STUN_URLS="${WEBRTC_STUN_URLS:-}"
if [[ -z "${WEBRTC_SERVER_TURN_URLS:-}" ]]; then
  if env_enabled "$WEBRTC_USE_LOCAL_TURN"; then
    export WEBRTC_SERVER_TURN_URLS="turn:127.0.0.1:$TURN_LISTEN_PORT?transport=tcp"
  else
    export WEBRTC_SERVER_TURN_URLS="$WEBRTC_TURN_URLS"
  fi
fi
export WEBRTC_TURN_USER="${WEBRTC_TURN_USER:-$TURN_USER}"
export WEBRTC_TURN_PASS="${WEBRTC_TURN_PASS:-$TURN_PASS}"
export WEBRTC_SYNC_MODE="${WEBRTC_SYNC_MODE:-strict_fifo}"
export WEBRTC_VIDEO_PREBUFFER_SECONDS="${WEBRTC_VIDEO_PREBUFFER_SECONDS:-2.0}"
export WEBRTC_AUDIO_PREBUFFER_SECONDS="${WEBRTC_AUDIO_PREBUFFER_SECONDS:-0.0}"
export WEBRTC_ADAPTIVE_FPS="${WEBRTC_ADAPTIVE_FPS:-0}"

log "Starting API with WebRTC relay policy"
log "TURN env file=$ENV_FILE"
log "WEBRTC_ICE_TRANSPORT_POLICY=$WEBRTC_ICE_TRANSPORT_POLICY"
log "WEBRTC_TURN_URLS=$WEBRTC_TURN_URLS"
log "WEBRTC_SERVER_TURN_URLS=$WEBRTC_SERVER_TURN_URLS"
log "WEBRTC_USE_LOCAL_TURN=$WEBRTC_USE_LOCAL_TURN"
log "WEBRTC_SYNC_MODE=$WEBRTC_SYNC_MODE"
log "WEBRTC_VIDEO_PREBUFFER_SECONDS=$WEBRTC_VIDEO_PREBUFFER_SECONDS"
log "WEBRTC_ADAPTIVE_FPS=$WEBRTC_ADAPTIVE_FPS"

exec bash "$REPO_ROOT/scripts/run_trt_stagewise_server.sh" "$@"
