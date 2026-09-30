#!/usr/bin/env bash
# Step 0b: fetch the reproduction inputs that git does not hold, from checksum-pinned S3 bundles (published 2026-09-30).
#   scripts/repro_400fps/05_fetch_inputs.sh [--engines]
#     - the six prepared harness avatars -> /workspace/experiments/avatar_diversity_20260927 (fixed path, see lib.sh)
#     - the UNet capture corpus          -> calibration/unet_multi_avatar_20260928 (also inside the r5 engine bundle)
#     - --engines: the published r5 + r2 engine sets and the TensorRT TAESD engine (the recipe r5 bundle,
#       configs/trt_bundles/rtx4070super-r5-srcg50-int8.json; RTX 4070 SUPER + TensorRT 10.3 only). A box booted with
#       MUSETALK_RECIPE=r5 already has them (scripts/vast_onstart.sh).
# Needs TRT_ARTIFACT_S3_BUCKET (or AVATAR_S3_BUCKET) and s3:GetObject on trt-artifacts/*: the runtime secret has both
# (set -a; . /workspace/.musetalk-runtime.env; set +a). CPU, disk and network only; safe to re-run. Each bundle is
# skipped when its stamp in .runtime/trt_artifacts/<name>/ still verifies; files already present (built or copied
# here) are adopted, i.e. verified against the bundle's manifest without downloading the payload. Present files that
# differ from the bundle are never overwritten unless MUSETALK_REPRO_FETCH_OVERWRITE=1.
source "$(dirname "$0")/lib.sh"

ENGINES=0
[ "${1:-}" = "--engines" ] && ENGINES=1
BUCKET="${TRT_ARTIFACT_S3_BUCKET:-${AVATAR_S3_BUCKET:-}}"
[ -n "$BUCKET" ] || die "set TRT_ARTIFACT_S3_BUCKET (e.g. set -a; . /workspace/.musetalk-runtime.env; set +a)"
STAGE="${MUSETALK_TRT_ARTIFACT_STAGE_DIR:-$REPO/tmp/trt_artifact_stage}"

# fetch NAME ROOT KEY SHA PROBE: bundle NAME restored into ROOT; PROBE (relative to ROOT) exists when a payload is there
fetch() {
  local name="$1" root="$2" key="$3" sha="$4" probe="$5" side="$REPO/.runtime/trt_artifacts/$1"
  local -a common=(--repo-root "$root" --strict --sidecar-dir "$side")
  mkdir -p "$root" "$STAGE"
  if [ ! -f "$side/.musetalk_trt_artifact_restored.json" ] && [ -e "$root/$probe" ]; then
    log "$name: $root/$probe exists; adopting (verify against the bundle manifest, no payload download)"
    if "$PY" -B scripts/trt_artifact_bundle.py "${common[@]}" adopt --uri "s3://$BUCKET/$key" --expected-sha256 "$sha"; then
      return 0
    fi
    [ "${MUSETALK_REPRO_FETCH_OVERWRITE:-0}" = 1 ] || die "$name: files under $root/$probe differ from the published \
bundle; move them aside (or MUSETALK_REPRO_FETCH_OVERWRITE=1 to restore over them) and re-run"
    log "$name: local files differ from the bundle; restoring it over them (MUSETALK_REPRO_FETCH_OVERWRITE=1)"
  fi
  log "$name: restore -> $root"
  "$PY" -B scripts/trt_artifact_bundle.py "${common[@]}" restore --uri "s3://$BUCKET/$key" --expected-sha256 "$sha" \
    --stage-dir "$STAGE" --skip-if-verified || die "$name: restore failed"
}

fetch repro-avatar-diversity-20260927 /workspace/experiments \
  trt-artifacts/repro-inputs/avatar-diversity-20260927/sha256-4fbb421484b119814c52ed40840ae41c0481086a53960847b9e215e01b015149/musetalk-repro-avatar-diversity-20260927.tar.gz \
  4fbb421484b119814c52ed40840ae41c0481086a53960847b9e215e01b015149 avatar_diversity_20260927
fetch repro-calibration-unet-multi-avatar-20260928 "$REPO" \
  trt-artifacts/repro-inputs/unet-multi-avatar-calibration-20260928/sha256-5b38ed6d0d776b43d405d85839cabeeaf143972ea67c56dd4eaac7e46bab3ded/musetalk-repro-calibration-unet-multi-avatar-20260928.tar.gz \
  5b38ed6d0d776b43d405d85839cabeeaf143972ea67c56dd4eaac7e46bab3ded "$CORPUS"

if [ "$ENGINES" = 1 ]; then
  DESC=configs/trt_bundles/rtx4070super-r5-srcg50-int8.json
  read -r KEY SHA SIDE WANT < <("$PY" -B -c 'import json,sys; d=json.load(open(sys.argv[1]))
print(d["s3_key"], d["sha256"], d["sidecar_dir"], d["host"]["engine_key"])' "$DESC") || die "cannot read $DESC"
  HAVE="$("$PY" -B scripts/musetalk_engine_keys.py key --kind unet_stagewise --venv "$(dirname "$(dirname "$PY")")" 2>/dev/null)"
  [ "$HAVE" = "$WANT" ] || die "the r5 engines are built for $WANT; this host is ${HAVE:-unknown}. Build them instead: 10_build_engines.sh"
  # same sidecar dir as scripts/vast_onstart.sh, so the resolver's bundle: prerequisite sees this restore
  fetch "$(basename "$SIDE")" "$REPO" "$KEY" "$SHA" "$PUBLISHED_R5/bs16/manifest.json"
fi
log "inputs ready; next: scripts/repro_400fps/00_check.sh"
