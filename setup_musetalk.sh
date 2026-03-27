#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf '[setup_musetalk.sh] This wrapper now targets the current TRT-stagewise server env at /content/py310_trt_exp.\n'
printf '[setup_musetalk.sh] Delegating to scripts/setup_trt_stagewise_server_env.sh\n'

exec bash "$SCRIPT_DIR/scripts/setup_trt_stagewise_server_env.sh" "$@"
