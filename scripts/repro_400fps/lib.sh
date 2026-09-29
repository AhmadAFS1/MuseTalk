# Shared settings for scripts/repro_400fps/*.sh. Sourced, never run directly.
# Every GPU, model or RAM-heavy step goes through scripts/box_guard.sh (GPU lease, RAM watchdog, pause file),
# because this box is shared with other sessions and the user's own servers.
#
# Layout is FIXED (the harness, quality tool and chin workflow hard-code these paths): the repo can live anywhere,
# but the main venv, the FaceMesh venv, the six prepared avatars and a writable /workspace must be where they are
# named below. Only the engine root, the output dir and the tag are configurable.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PKG="$REPO/scripts/repro_400fps"
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python          # torch 2.5.1+cu121, TensorRT 10.3, modelopt, diffusers
FACEMESH_PY=/workspace/SoulX-FlashHead/.venv/bin/python          # mediapipe 0.10.9 (chin tracker; hard-coded in the workflow)
ACCEPTED=/workspace/experiments/avatar_diversity_20260927        # the six prepared avatars (hard-coded in the harness)
CORPUS=calibration/unet_multi_avatar_20260928                    # UNet capture corpus, 352 main + 96 holdout bs8 files
RECIPE="${MUSETALK_REPRO_RECIPE:-docs/fps_comparisons/4070s_400fps_20260928/int8_study/recipe_gmac_0.50.json}"  # builder only (--set r5)
PUBLISHED_R5=models/tensorrt_unet_stagewise_sm89_srcg50
PUBLISHED_R2=models/tensorrt_unet_stagewise_sm89_srcmix
RUNTIME_ENV=.runtime/musetalk_trt_local_sm89.env                 # read by every harness run (scripts/chin_multistream/gpu.py)
ROOT_NAME="${MUSETALK_REPRO_ROOT:-tensorrt_unet_stagewise_sm89_r5}"
ENGINE_ROOT="models/$ROOT_NAME"
OUT="${MUSETALK_REPRO_OUT:-docs/fps_comparisons/repro_400fps}"
QRUNS_OUT="$OUT/quality_runs"
PUBLISHED_QRUNS=docs/fps_comparisons/4070s_300fps_impl_20260928/quality_metrics/runs
GUARD="scripts/box_guard.sh run --wait-min 120"
mkdir -p "$OUT" "$QRUNS_OUT"
# knobs that change engine builds or the TAESD engine key without being recorded; the package runs with them unset
for v in MUSETALK_TRT_AVG_TIMING_ITERS MUSETALK_TAESD_TRT_OPT_LEVEL MUSETALK_TAESD_TRT_STRONGLY_TYPED MUSETALK_TAESD_TRT_DIR; do
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
  $GUARD --min-avail-gb "$gb" --label "repro_$label" -- "$@" > "$OUT/$label.log" 2>&1
  local rc=$?
  log "$label rc=$rc"
  return $rc
}
# harness <label> <engine root> <min_avail_gb> <extra args...>: the six-avatar full-recipe multi-stream harness
# (TensorRT TAESD, 100% chin, refined seam, unchanged chin.py) with the given UNet engine set
harness() {
  local label=$1 root=$2 gb=$3; shift 3
  ensure_runtime_env
  guarded "$label" "$gb" "$PY" scripts/chin_multistream_render.py --backend stagewise16_taesdtrt \
    --flags "MUSETALK_UNET_STAGEWISE_CACHE_DIR=$root" --out-root "$OUT/chin_multistream" --label "$label" "$@"
}
summary() {  # summary <label>: status + the harness SUMMARY fields that matter
  local s; s=$(grep -a '^SUMMARY' "$OUT/$1.log" | tail -1)
  [ -n "$s" ] || { echo "NO SUMMARY (failed; see $OUT/$1.log)"; return 1; }
  echo "$(echo "$s" | awk '{print $3}') $(echo "$s" | grep -o '"aggregate_fps_per_repeat": \[[^]]*\]\|"median_aggregate_fps": [0-9.]*\|"all_repeats_ge_min_timed_s": [a-z]*\|"deterministic_per_identity": [a-z]*\|"all_clips_match_accepted": [a-z]*' | tr '\n' ' ')"
}
