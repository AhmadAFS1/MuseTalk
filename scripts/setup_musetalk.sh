#!/usr/bin/env bash
# Compatibility shim: same as <repo>/setup_musetalk.sh (legacy flags -> scripts/install_musetalk.sh).
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../setup_musetalk.sh" "$@"
