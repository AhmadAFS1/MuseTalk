#!/usr/bin/env bash
# Vast headless/args entrypoint for the prebuilt image, NOT a source installer.
# The host pulls the pinned image before this script can run. Registry login
# belongs in Vast image_login, never in this script or the container environment.
set +x
set -Eeuo pipefail
umask 077

case "${1:-serve}" in
  serve|onstart|check) ;;
  *) echo 'VAST_DOCKER FAILED: expected serve, onstart, or check' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo 'VAST_DOCKER FAILED: unexpected arguments' >&2; exit 2; }

# Fixed image paths: /workspace is mutable and may be a provider volume.
# Never substitute a host checkout, download code, repair packages or engines.
for required in /opt/musetalk/release.json \
                /opt/musetalk/app/docker/musetalk/entrypoint.sh \
                /opt/musetalk/app/docker/musetalk/supervise.py \
                /opt/musetalk/app/scripts/vast_onstart.sh; do
  [[ -f "$required" && -r "$required" && ! -L "$required" ]] || {
    echo 'VAST_DOCKER FAILED: required baked runtime file missing or symlinked' >&2
    exit 1
  }
done
[[ -x /opt/musetalk/venv/bin/python ]] || {
  echo 'VAST_DOCKER FAILED: baked Python environment unavailable' >&2
  exit 1
}

# Do not call this READY: verification, private models, GPU probes and avatar
# warming still happen in the canonical lifecycle. Measure provider pull time
# separately from the scale-up request timestamp; this marker starts after pull.
printf 'VAST_DOCKER ENTRYPOINT: utc=%s unix_seconds=%s action=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$(date +%s)" "${1:-serve}"
exec /bin/bash /opt/musetalk/app/docker/musetalk/entrypoint.sh "${1:-serve}"
