#!/usr/bin/env bash
set -euo pipefail

echo "Installing WebRTC dependencies for local testing..."
python -m pip install --upgrade pip
python -m pip install aiortc av aioice

echo "Done. Restart the API server to load WebRTC modules."
