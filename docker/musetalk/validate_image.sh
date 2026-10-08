#!/usr/bin/env bash
# Read-only image diagnostics + isolated GPU-less import check. Never publishes.
set -Eeuo pipefail
image="${1:?Usage: validate_image.sh IMAGE@sha256:DIGEST OUTPUT_DIRECTORY}"
out="${2:?Output directory required}"
[[ "$image" =~ @sha256:[0-9a-f]{64}$ ]] || { echo "Immutable digest required" >&2; exit 2; }
[[ ! -e "$out" ]] || { echo "Output already exists" >&2; exit 2; }
mkdir -p "$out"
docker pull --platform linux/amd64 "$image"
docker image inspect "$image" > "$out/image-inspect.json"
docker history --no-trunc --format '{{json .}}' "$image" > "$out/image-history.jsonl"
docker run --rm --platform linux/amd64 --network none --entrypoint /bin/bash "$image" \
  /opt/musetalk/app/docker/musetalk/entrypoint.sh check > "$out/cpu-check.log" 2>&1
# Human/agent layer and license review is still required. This check does not
# substitute for GPU performance, quality, startup, or external WebRTC evidence.
printf '%s\n' "CPU-only image check passed. GPU and fresh-instance release gates remain separate."
