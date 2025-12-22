#!/usr/bin/env bash
# Helper to start coturn with the project config.
# Usage: sudo ./scripts/run_turnserver.sh

set -euo pipefail

CONFIG="${CONFIG:-/workspace/turnserver.conf}"

if ! command -v turnserver >/dev/null 2>&1; then
  echo "turnserver not found. Install coturn first." >&2
  exit 1
fi

echo "Starting turnserver with config: ${CONFIG}"
exec sudo turnserver -c "${CONFIG}"
