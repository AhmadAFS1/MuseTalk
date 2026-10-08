#!/usr/bin/env bash
# Container lifecycle wrapper. Server startup/verification/drain remain canonical.
set -Eeuo pipefail
umask 077
export REPO_ROOT=/opt/musetalk/app
export VENV_PATH=/opt/musetalk/venv
export MUSETALK_INSTALL_STATE_FILE=/opt/musetalk/install_state.json
export WEBRTC_NATIVE_VP8_DIR=/opt/musetalk/native_vp8
export MUSETALK_RELEASE_MANIFEST=/opt/musetalk/release.json
export MUSETALK_IMMUTABLE_RUNTIME=1
export AUTO_SETUP=0 SETUP_CLEAN=0 SETUP_SELFTEST=0
export WORKSPACE="${MUSETALK_STATE_DIR:-/workspace/musetalk-runtime}"
[[ "$WORKSPACE" == /* ]] || { echo "MUSETALK_STATE_DIR must be absolute" >&2; exit 2; }
export LOG_DIR="$WORKSPACE/logs" ONSTART_LOG="$WORKSPACE/onstart.log"
export PORT="${PORT:-8000}"
export TURN_ENV_FILE="$WORKSPACE/turn.env"
export LINGUA_CONTROL_PLANE_ENV_FILE=/dev/null
export MUSETALK_ENV_OVERRIDES_FILE=""
unset MUSETALK_HOST_FACTS_JSON MUSETALK_NVIDIA_SMI MUSETALK_RESOLVER MUSETALK_RESOLVER_PYTHON
unset MUSETALK_ENGINE_STORE_TOOL MUSETALK_SERVER_LAUNCHER MUSETALK_INSTALLER
export HF_HOME="$REPO_ROOT/models/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LINGUA_WORKER_BUILD
LINGUA_WORKER_BUILD="$(/opt/musetalk/venv/bin/python "$REPO_ROOT/docker/musetalk/release.py" revision \
  --manifest "$MUSETALK_RELEASE_MANIFEST")"
mkdir -p "$WORKSPACE" "$LOG_DIR"
cd "$REPO_ROOT"

case "${1:-serve}" in
  check) exec /opt/musetalk/venv/bin/python docker/musetalk/release.py cpu-check \
    --root "$REPO_ROOT" --manifest "$MUSETALK_RELEASE_MANIFEST" ;;
  serve|onstart) ;;
  *) echo "Expected serve, onstart, or check" >&2; exit 2 ;;
esac

# Vast's SSH/Jupyter mode invokes `onstart`; Docker ENTRYPOINT invokes `serve`.
# Serialize the whole supervisor lifetime to prevent duplicate API/TURN ownership.
exec 9>"$WORKSPACE/container.lock"
flock -n 9 || { echo "MuseTalk lifecycle already running" >&2; exit 1; }
for name in uploads results; do
  mkdir -p "$WORKSPACE/$name"
  if [[ -L "$REPO_ROOT/$name" ]]; then
    [[ "$(readlink "$REPO_ROOT/$name")" == "$WORKSPACE/$name" ]] || exit 1
  elif [[ -e "$REPO_ROOT/$name" ]]; then
    echo "Refusing to replace existing image path: $name" >&2; exit 1
  else
    ln -s "$WORKSPACE/$name" "$REPO_ROOT/$name"
  fi
done

# The Python owner handles TERM while bootstrap is running, unlike a shell
# blocked on a foreground command. Descriptor 9 retains lifecycle ownership.
exec /opt/musetalk/venv/bin/python "$REPO_ROOT/docker/musetalk/supervise.py"
