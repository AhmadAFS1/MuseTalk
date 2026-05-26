#!/usr/bin/env bash
# Compatibility helper for the older WebRTC start path.
#
# Do not edit stale TURN IPs into this file. Put current TURN settings in
# .env.webrtc-turn.local or pass WEBRTC_TURN_URLS/WEBRTC_TURN_PASS directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

export WEBRTC_RELAY_ENABLED="${WEBRTC_RELAY_ENABLED:-1}"
export WEBRTC_ICE_TRANSPORT_POLICY="${WEBRTC_ICE_TRANSPORT_POLICY:-relay}"

echo "Starting MuseTalk through the WebRTC relay launcher"
echo "  env file: ${TURN_ENV_FILE:-$REPO_ROOT/.env.webrtc-turn.local}"
echo "  pass extra TRT server args after this script, for example: --profile throughput_record"

exec bash "$REPO_ROOT/scripts/run_webrtc_relay_api_server.sh" "$@"
