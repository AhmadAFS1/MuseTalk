#!/usr/bin/env bash
# Paste into Vast On-start ONLY for the full digest-pinned MuseTalk image.
# Vast pulls the image using image_login before this script starts. No docker
# daemon, registry token, git checkout, dependency install or build is used here.
set +x
set -Eeuo pipefail
umask 077

case "${1:-onstart}" in
  serve|onstart|check) ;;
  *) echo '[docker-bootstrap] expected serve, onstart, or check' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo '[docker-bootstrap] unexpected arguments' >&2; exit 2; }

mkdir -p /workspace
BOOT_LOG=/workspace/bootstrap.log
exec > >(tee -a "$BOOT_LOG") 2>&1
trap 'rc=$?; printf "[docker-bootstrap] failed at line %s with exit %s\n" "$LINENO" "$rc"; exit "$rc"' ERR

START_SCRIPT=/opt/musetalk/app/scripts/vast_docker_onstart.sh
[[ -f /opt/musetalk/release.json && ! -L /opt/musetalk/release.json &&
   -f "$START_SCRIPT" && ! -L "$START_SCRIPT" &&
   -x /opt/musetalk/venv/bin/python ]] || {
  echo '[docker-bootstrap] full MuseTalk image required; no install fallback' >&2
  exit 1
}
if [[ "${1:-onstart}" == check ]]; then
  exec /bin/bash "$START_SCRIPT" check
fi

# EC2 reads the runtime secret with its IAM role and injects the worker's values
# at creation. Its role is not inherited by Vast. Never put AWS keys in this file.
# For an independently authorized worker secret-reader, set the alternate mode
# and inject that identity outside the script instead.
case "${MUSETALK_RUNTIME_CONFIG_SOURCE:-injected}" in
  injected)
    [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]] || {
      echo '[docker-bootstrap] EC2-injected worker runtime credentials missing' >&2
      exit 1
    }
    unset MUSETALK_AWS_SECRET_ID
    ;;
  secretsmanager)
    export MUSETALK_AWS_SECRET_ID="${MUSETALK_AWS_SECRET_ID:-arn:aws:secretsmanager:us-east-1:211125449207:secret:lingua/musetalk-worker-runtime-Dof4b8}"
    export MUSETALK_AWS_SECRET_REGION="${MUSETALK_AWS_SECRET_REGION:-us-east-1}"
    ;;
  *) echo '[docker-bootstrap] invalid runtime configuration source' >&2; exit 2 ;;
esac

export MUSETALK_SECRETS_STRICT=true MUSETALK_SECRETS_VERIFY_S3=1
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_REGION="${AWS_REGION:-$AWS_DEFAULT_REGION}"
export PORT="${PORT:-8000}" STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-1800}"
export AUTO_SETUP=0 SETUP_CLEAN=0 SETUP_SELFTEST=0 SETUP_FULL_STACK=1

printf '[docker-bootstrap] start %s; baked runtime; port=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$PORT"
# The baked manifest pins the native 3090 r5 configuration and verifies all files.
# The canonical supervisor owns its lifecycle lock, API/TURN and graceful exit.
exec /bin/bash "$START_SCRIPT" "${1:-onstart}"
