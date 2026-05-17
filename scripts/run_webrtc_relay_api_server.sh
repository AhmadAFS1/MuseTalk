#!/usr/bin/env bash
# Launch the current TRT-stagewise API with WebRTC forced through TURN relay.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_FILE="${TURN_ENV_FILE:-$REPO_ROOT/.env.webrtc-turn.local}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

TURN_PUBLIC_IP="${TURN_PUBLIC_IP:-}"
TURN_PUBLIC_PORT="${TURN_PUBLIC_PORT:-${TURN_LISTEN_PORT:-3478}}"
TURN_LISTEN_PORT="${TURN_LISTEN_PORT:-3478}"
TURN_USER="${TURN_USER:-webrtc}"
TURN_PASS="${TURN_PASS:-${WEBRTC_TURN_PASS:-}}"

if [[ -z "$TURN_PUBLIC_IP" ]]; then
  echo "TURN_PUBLIC_IP is required. Set it in $ENV_FILE." >&2
  exit 1
fi

if [[ -z "$TURN_PASS" ]]; then
  echo "TURN_PASS is required. Set it in $ENV_FILE." >&2
  exit 1
fi

export WEBRTC_ICE_TRANSPORT_POLICY="${WEBRTC_ICE_TRANSPORT_POLICY:-relay}"
export WEBRTC_STUN_URLS="${WEBRTC_STUN_URLS:-}"
export WEBRTC_TURN_URLS="${WEBRTC_TURN_URLS:-turn:$TURN_PUBLIC_IP:$TURN_PUBLIC_PORT?transport=tcp}"
export WEBRTC_SERVER_TURN_URLS="${WEBRTC_SERVER_TURN_URLS:-turn:127.0.0.1:$TURN_LISTEN_PORT?transport=tcp}"
export WEBRTC_TURN_USER="${WEBRTC_TURN_USER:-$TURN_USER}"
export WEBRTC_TURN_PASS="${WEBRTC_TURN_PASS:-$TURN_PASS}"
export WEBRTC_VIDEO_PREBUFFER_SECONDS="${WEBRTC_VIDEO_PREBUFFER_SECONDS:-0}"
export WEBRTC_AUDIO_PREBUFFER_SECONDS="${WEBRTC_AUDIO_PREBUFFER_SECONDS:-0.15}"

echo "Starting API with WebRTC relay policy"
echo "  WEBRTC_ICE_TRANSPORT_POLICY=$WEBRTC_ICE_TRANSPORT_POLICY"
echo "  WEBRTC_TURN_URLS=$WEBRTC_TURN_URLS"
echo "  WEBRTC_SERVER_TURN_URLS=$WEBRTC_SERVER_TURN_URLS"

exec bash "$REPO_ROOT/scripts/run_trt_stagewise_server.sh" "$@"
