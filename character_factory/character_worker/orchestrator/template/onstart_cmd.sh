#!/usr/bin/env bash
# Vast template "on-start" field for the character-creation worker.
# Clones (or refreshes) MuseTalk, then hands over to character_factory/character_worker/vast_onstart_character.sh,
# which installs everything idempotently and starts the worker API on $CHARACTER_WORKER_PORT (default 8010).
set -Eeuo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
export WORKSPACE
mkdir -p "$WORKSPACE"
exec > >(tee -a "$WORKSPACE/character_bootstrap.log") 2>&1
echo "[bootstrap] $(date -u '+%Y-%m-%d %H:%M:%S UTC') start"
REPO_URL="${MUSETALK_REPO_URL:-https://github.com/AhmadAFS1/MuseTalk.git}"
REF="${MUSETALK_REF:-main}"
REPO_DIR="$WORKSPACE/MuseTalk"
if ! command -v git >/dev/null 2>&1; then apt-get update -y && apt-get install -y git ca-certificates; fi
git config --global --add safe.directory "$REPO_DIR" || true
if [[ ! -d "$REPO_DIR/.git" ]]; then
  # --branch accepts branch/tag names, not an immutable commit SHA. Fetching into a
  # staging checkout supports both and never leaves a half-created target behind.
  if [[ -e "$REPO_DIR" ]] && ! rmdir "$REPO_DIR"; then
    echo "[bootstrap] refusing to replace a nonempty checkout without .git: $REPO_DIR" >&2
    exit 1
  fi
  stage="$(mktemp -d "$WORKSPACE/.character-checkout.XXXXXX")"
  trap 'rm -rf "$stage"' EXIT
  git init -q "$stage"
  git -C "$stage" remote add origin "$REPO_URL"
  git -C "$stage" fetch --depth 1 origin "$REF"
  git -C "$stage" checkout -q --detach FETCH_HEAD
  mv "$stage" "$REPO_DIR"
  trap - EXIT
else
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
  git -C "$REPO_DIR" fetch --depth 1 origin "$REF"
  git -C "$REPO_DIR" checkout -q --detach FETCH_HEAD
fi
echo "[bootstrap] MuseTalk at $(git -C "$REPO_DIR" rev-parse --short HEAD)"
exec bash "$REPO_DIR/character_factory/character_worker/vast_onstart_character.sh"
