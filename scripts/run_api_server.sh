#!/usr/bin/env bash
# Helper to run the MuseTalk API with WebRTC/TURN env vars set.
# Fill in your TURN creds and endpoints once, then re-run this script.

set -euo pipefail

# ---- Edit these for your deployment ----
export WEBRTC_TURN_URLS="${WEBRTC_TURN_URLS:-turn:195.142.145.66:12885?transport=udp,turn:195.142.145.66:12964?transport=tcp}"
export WEBRTC_TURN_USER="${WEBRTC_TURN_USER:-webrtc}"
export WEBRTC_TURN_PASS="${WEBRTC_TURN_PASS:-CHANGE_THIS_PASSWORD}"
# Uncomment to force relay-only testing (no STUN):
# export WEBRTC_STUN_URLS=""

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting api_server.py on ${HOST}:${PORT} with TURN URLs: ${WEBRTC_TURN_URLS}"
exec python api_server.py --host "${HOST}" --port "${PORT}"
