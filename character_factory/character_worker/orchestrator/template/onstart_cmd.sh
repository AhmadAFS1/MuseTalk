#!/usr/bin/env bash
# Vast template "on-start" field for the character-creation worker.
# Clones (or refreshes) MuseTalk, then hands over to character_factory/character_worker/vast_onstart_character.sh,
# which installs everything idempotently and starts the worker API on $CHARACTER_WORKER_PORT (default 8010).
set -Eeuo pipefail
mkdir -p /workspace
exec > >(tee -a /workspace/character_bootstrap.log) 2>&1
echo "[bootstrap] $(date -u '+%Y-%m-%d %H:%M:%S UTC') start"
REPO_URL="${MUSETALK_REPO_URL:-https://github.com/AhmadAFS1/MuseTalk.git}"
REF="${MUSETALK_REF:-main}"
if ! command -v git >/dev/null 2>&1; then apt-get update -y && apt-get install -y git ca-certificates; fi
if [ ! -d /workspace/MuseTalk/.git ]; then
  git clone --depth 1 --branch "$REF" "$REPO_URL" /workspace/MuseTalk
else
  git -C /workspace/MuseTalk fetch --depth 1 origin "$REF" && git -C /workspace/MuseTalk checkout -q FETCH_HEAD
fi
git config --global --add safe.directory /workspace/MuseTalk || true
echo "[bootstrap] MuseTalk at $(git -C /workspace/MuseTalk rev-parse --short HEAD)"
exec bash /workspace/MuseTalk/character_factory/character_worker/vast_onstart_character.sh
