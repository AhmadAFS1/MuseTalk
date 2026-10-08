# Shared settings for scripts/repro_400fps/*.sh. Sourced, never run directly.
# Every GPU, model or RAM-heavy step goes through scripts/box_guard.sh (GPU lease, RAM watchdog, pause file),
# because this box is shared with other sessions and the user's own servers.
#
# Historical defaults are retained. MUSETALK_REPRO_* overrides relocate the main venv,
# corpus, accepted fixtures, comparison root and explicit runtime profile. The frozen
# chin Tracker still expects SoulX-FlashHead/.venv under MUSETALK_REPRO_WORKSPACE.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PKG="$REPO/scripts/repro_400fps"
PY="${MUSETALK_REPRO_PYTHON:-/workspace/.venvs/musetalk_trt_stagewise/bin/python}"
FACEMESH_PY="${MUSETALK_REPRO_WORKSPACE:-/workspace}/SoulX-FlashHead/.venv/bin/python"
ACCEPTED="${MUSETALK_REPRO_ACCEPTED:-/workspace/experiments/avatar_diversity_20260927}"
CORPUS="${MUSETALK_REPRO_CORPUS:-calibration/unet_multi_avatar_20260928}"
RECIPE="${MUSETALK_REPRO_RECIPE:-docs/fps_comparisons/4070s_400fps_20260928/int8_study/recipe_gmac_0.50.json}"  # builder only (--set r5)
PUBLISHED_R5="${MUSETALK_REPRO_COMPARISON_ROOT:-models/tensorrt_unet_stagewise_sm89_srcg50}"
PUBLISHED_R2="${MUSETALK_REPRO_R2_ROOT:-models/tensorrt_unet_stagewise_sm89_srcmix}"
RUNTIME_ENV="${MUSETALK_REPRO_RUNTIME_ENV:-.runtime/musetalk_trt_local_sm89.env}"
export MUSETALK_REPRO_RUNTIME_ENV="$RUNTIME_ENV"
ROOT_NAME="${MUSETALK_REPRO_ROOT:-tensorrt_unet_stagewise_sm89_r5}"
ENGINE_ROOT="models/$ROOT_NAME"
OUT="${MUSETALK_REPRO_OUT:-docs/fps_comparisons/repro_400fps}"
QRUNS_OUT="$OUT/quality_runs"
PUBLISHED_QRUNS=docs/fps_comparisons/4070s_300fps_impl_20260928/quality_metrics/runs
GUARD="scripts/box_guard.sh run --wait-min 120"
mkdir -p "$OUT" "$QRUNS_OUT"
# knobs that change engine builds or the TAESD engine key without being recorded; the package runs with them unset
for v in MUSETALK_TRT_AVG_TIMING_ITERS MUSETALK_TAESD_TRT_OPT_LEVEL MUSETALK_TAESD_TRT_STRONGLY_TYPED MUSETALK_TAESD_TRT_DIR \
         MUSETALK_TAESD_TRT_HW_COMPAT; do
  [ "${MUSETALK_REPRO_EXPLICIT_PROFILE:-0}" = 1 ] && continue
  [ -n "${!v:-}" ] && echo "[repro] note: unsetting $v=${!v} (it would change the engines)" >&2
  unset "$v"
done

log() { printf '[repro %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "ERROR: $*"; exit 1; }
# tag <engine root>: the label a root's runs are stored under. Published roots are prefixed "repro_" so a re-measure
# never overwrites the published records (quality runs, captures) that the READMEs and sign-off videos read.
tag() { local t; t="${MUSETALK_REPRO_TAG:-$(basename "$1" | sed 's/tensorrt_unet_stagewise_sm89_//')}"; case "$t" in repro_*) echo "$t" ;; *) echo "repro_$t" ;; esac; }
# ensure_runtime_env: every harness run reads $RUNTIME_ENV; install the recorded one when it is absent
ensure_runtime_env() {
  if [ ! -e "$RUNTIME_ENV" ]; then
    [ "${MUSETALK_REPRO_EXPLICIT_PROFILE:-0}" = 1 ] && die "explicit runtime profile is missing: $RUNTIME_ENV"
    mkdir -p "$(dirname "$RUNTIME_ENV")"
    cp "$PKG/musetalk_trt_local_sm89.env" "$RUNTIME_ENV"
    log "installed $RUNTIME_ENV from the published runs' record"
  fi
}
# guarded <label> <min_avail_gb> <cmd...>: run one heavy step under the GPU lease; stdout/stderr go to $OUT/<label>.log.
# min_avail_gb is what the step needs free at start: box_guard's watchdog kills below 3 GB, so it is the step's own
# peak MemAvailable use + ~3 GB (INT8 build ~10.5 GB, N15 ~9.6, 6-stream harness ~7.7).
guarded() {
  local label=$1 gb=$2; shift 2
  log "$label: $* (log $OUT/$label.log)"
  local command=("$@")
  # Opt-in for new GPU builds: retain the canonical per-step lease, while the
  # 3090 watchdog rejects foreign GPU work that begins after lease acquisition.
  # Do not put a second box_guard around this script (nested leases deadlock).
  if [ "${MUSETALK_REPRO_GPU_WATCH:-0}" = 1 ]; then
    local watch="$REPO/scripts/repro_3090/watch.py"
    [ -f "$watch" ] || die "GPU watchdog missing: $watch"
    command=(python3 "$watch" --out "$OUT/$label.gpu_watch.jsonl" -- "${command[@]}")
  fi
  $GUARD --min-avail-gb "$gb" --label "repro_$label" -- "${command[@]}" > "$OUT/$label.log" 2>&1
  local rc=$?
  log "$label rc=$rc"
  return $rc
}
# hw_compat_of <engine root>: the set's TensorRT hardware compatibility level (none | ampere_plus). A hardware-compatible
# UNet set is gated and measured with the hardware-compatible TAESD engine (MUSETALK_TAESD_TRT_HW_COMPAT), a default
# set with the default one.
hw_compat_of() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("hardware_compatibility_level") or "none")' \
    "$1/bs16/manifest.json" 2>/dev/null || echo none
}
# harness <label> <engine root> <min_avail_gb> <extra args...>: the six-avatar full-recipe multi-stream harness
# (TensorRT TAESD, 100% chin, refined seam, unchanged chin.py) with the given UNet engine set
harness() {
  local label=$1 root=$2 gb=$3 flags hw; shift 3
  flags="MUSETALK_UNET_STAGEWISE_CACHE_DIR=$root"
  hw=$(hw_compat_of "$root"); [ "$hw" = none ] || flags="$flags,MUSETALK_TAESD_TRT_HW_COMPAT=$hw"
  ensure_runtime_env
  guarded "$label" "$gb" "$PY" scripts/chin_multistream_render.py --backend stagewise16_taesdtrt \
    --flags "$flags" --out-root "$OUT/chin_multistream" --label "$label" "$@"
}
summary() {  # summary <label>: status + the harness SUMMARY fields that matter
  local s; s=$(grep -a '^SUMMARY' "$OUT/$1.log" | tail -1)
  [ -n "$s" ] || { echo "NO SUMMARY (failed; see $OUT/$1.log)"; return 1; }
  echo "$(echo "$s" | awk '{print $3}') $(echo "$s" | grep -o '"aggregate_fps_per_repeat": \[[^]]*\]\|"median_aggregate_fps": [0-9.]*\|"all_repeats_ge_min_timed_s": [a-z]*\|"deterministic_per_identity": [a-z]*\|"all_clips_match_accepted": [a-z]*' | tr '\n' ' ')"
}
