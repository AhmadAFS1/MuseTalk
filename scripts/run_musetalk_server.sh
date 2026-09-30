#!/usr/bin/env bash
# MuseTalk API launcher (startup rework 2026-09-28; operator guide: docs/STARTUP.md).
#
# Boots the recipe that scripts/musetalk_host_profile.py resolves for THIS host:
#   r5          (default) fast + configs/recipes/r5.env: the ~400 fps r5 engines from the pinned S3
#               bundle (one TensorRT set for every Ampere-or-newer GPU, restored by vast_onstart.sh)
#               + the live-tested serving levers; on other GPUs the fast engines + those levers
#   fast        compiled TAESD + validated torch_tensorrt bs8 .ts UNet, else eager UNet
#   fast300     fast + the 300 fps levers of configs/recipes/fast300.env whose gates pass here
#   legacy_int8 exec scripts/run_trt_stagewise_server.sh unchanged (one-line rollback)
#
# Layering, highest wins: caller env > overrides files (MUSETALK_ENV_OVERRIDES_FILE, colon list,
# default .runtime/musetalk_overrides.env) > .runtime/musetalk_resolved.env (rewritten by the
# resolver on EVERY launch) > code defaults. Files are parsed, never sourced; a key is exported
# only while unset (scripts/lib/musetalk_env_layers.sh). Unknown knobs pass through untouched.
#
# This script never imports torch. The only Python it runs is the stdlib-only resolver, the
# CPU-only native VP8 preflight (CUDA hidden) and a stdlib JSON writer.

set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_START_TS="$(date +%s)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE:-}"
if [[ -z "$WORKSPACE_ROOT" ]]; then
  if [[ "$REPO_ROOT" == /workspace/* || "$REPO_ROOT" == "/workspace" ]]; then
    WORKSPACE_ROOT="/workspace"
  elif [[ "$REPO_ROOT" == /content/* || "$REPO_ROOT" == "/content" ]]; then
    WORKSPACE_ROOT="/content"
  else
    WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
  fi
fi
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/.venvs/musetalk_trt_stagewise}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LAUNCH_PROFILE_ARG=""
VALIDATE_ONLY=0
PRINT_ENV=0
ORIG_ARGS=("$@")
LAUNCH_TMP_DIR=""
EXPECT_VAE="any"
EXPECT_UNET="any"

log() {
  # --print-env keeps stdout machine-readable: logs go to stderr there.
  if (( PRINT_ENV )); then
    printf '[%s] %s\n' "$SCRIPT_NAME" "$*" >&2
  else
    printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
  fi
}

warn() {
  printf '[%s] WARNING: %s\n' "$SCRIPT_NAME" "$*" >&2
  LAUNCH_WARNINGS+=("$*")
}

die() {
  printf '[%s] ERROR: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

declare -a LAUNCH_WARNINGS=()

cleanup_tmp() {
  if [[ -n "$LAUNCH_TMP_DIR" && -d "$LAUNCH_TMP_DIR" ]]; then
    rm -rf "$LAUNCH_TMP_DIR"
  fi
}
trap cleanup_tmp EXIT

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options]

Launch the MuseTalk API with the recipe resolved for this host (docs/STARTUP.md).

Options:
  --host HOST        Bind host (default: $HOST)
  --port PORT        Bind port (default: $PORT)
  --venv-path PATH   Python venv path (default: $VENV_PATH)
  --repo-root PATH   MuseTalk repo root (default: $REPO_ROOT)
  --profile NAME     Accepted for compatibility; logged, otherwise ignored (legacy_int8 uses it)
  --validate-only    Resolve + preflight, then exit 0 without starting the API
  --print-env        Print the effective value and source of every managed key, then exit 0
                     (non-destructive: the resolver writes to a temp dir)
  --help             Show this help text

Environment (see configs/musetalk_overrides.env.example for every lever):
  MUSETALK_RECIPE=r5|fast|fast300|legacy_int8  recipe (default r5)
  MUSETALK_ENV_OVERRIDES_FILE=a.env:b.env    overrides files (default .runtime/musetalk_overrides.env)
  MUSETALK_UNET_MODE=auto|trt|eager          UNet selection (resolver)
  MUSETALK_VP8_FALLBACK=1                    native VP8 preflight failure -> pyav with a warning
  MUSETALK_RUNTIME_DIR                       default <repo>/.runtime
  MUSETALK_RESOLVED_ENV_FILE / MUSETALK_RESOLVED_REPORT_FILE / MUSETALK_LAUNCH_STATE_FILE
  MUSETALK_RESOLVER / MUSETALK_RESOLVER_PYTHON   resolver script / interpreter (default: venv python)
  MUSETALK_LAUNCHER_DRY_RUN=1                do everything, print the final exec, exit 0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      [[ $# -ge 2 ]] || die "--host requires a value"
      HOST="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port requires a value"
      PORT="$2"
      shift 2
      ;;
    --venv-path)
      [[ $# -ge 2 ]] || die "--venv-path requires a value"
      VENV_PATH="$2"
      shift 2
      ;;
    --repo-root)
      [[ $# -ge 2 ]] || die "--repo-root requires a value"
      REPO_ROOT="$2"
      shift 2
      ;;
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires a value"
      LAUNCH_PROFILE_ARG="$2"
      shift 2
      ;;
    --validate-only)
      VALIDATE_ONLY=1
      shift
      ;;
    --print-env)
      PRINT_ENV=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ -d "$REPO_ROOT" ]] || die "Repo root not found: $REPO_ROOT"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
VENV_PY="$VENV_PATH/bin/python"
[[ -x "$VENV_PY" ]] || die "Venv python not found: $VENV_PY (install with scripts/install_musetalk.sh)"
[[ -f "$REPO_ROOT/api_server.py" ]] || die "api_server.py not found under: $REPO_ROOT"

MT_ENV_LOG_PREFIX="$SCRIPT_NAME"
# shellcheck source=lib/musetalk_env_layers.sh
source "$SCRIPT_DIR/lib/musetalk_env_layers.sh"

RUNTIME_DIR="${MUSETALK_RUNTIME_DIR:-$REPO_ROOT/.runtime}"
RESOLVED_ENV="${MUSETALK_RESOLVED_ENV_FILE:-$RUNTIME_DIR/musetalk_resolved.env}"
RESOLVED_JSON="${MUSETALK_RESOLVED_REPORT_FILE:-$RUNTIME_DIR/musetalk_resolved.json}"
LAUNCH_STATE="${MUSETALK_LAUNCH_STATE_FILE:-$RUNTIME_DIR/musetalk_launch_${PORT}.json}"
RECIPE_FILES_DIR="$REPO_ROOT/configs/recipes"
RESOLVED_ENV_USED=""
RESOLVED_JSON_USED=""

if [[ -n "$LAUNCH_PROFILE_ARG" ]]; then
  log "--profile $LAUNCH_PROFILE_ARG accepted for compatibility (the resolver owns batch/worker sizing)"
fi

# ------------------------------------------------------------------ recipe dispatch
mt_env_effective_recipe "$REPO_ROOT"
RECIPE="$MT_ENV_RECIPE"
RECIPE_SOURCE="$MT_ENV_RECIPE_SOURCE"

dispatch_legacy() {
  local legacy="$REPO_ROOT/scripts/run_trt_stagewise_server.sh"
  local -a args=()
  local arg
  for arg in "${ORIG_ARGS[@]}"; do
    [[ "$arg" == "--print-env" ]] && continue
    args+=("$arg")
  done
  [[ -f "$legacy" ]] || die "legacy_int8 recipe requested but $legacy is missing"
  export MUSETALK_RECIPE=legacy_int8
  log "recipe=legacy_int8 (source=$RECIPE_SOURCE): handing over to the unchanged legacy chain"
  log "overrides files and the resolver are NOT applied to legacy_int8 (rollback = the old chain exactly)"
  if (( PRINT_ENV )); then
    printf "MUSETALK_RECIPE='legacy_int8'  # %s\n" "$RECIPE_SOURCE"
    printf '# exec: bash %s %s\n' "$legacy" "${args[*]}"
    exit 0
  fi
  if mt_env_is_true "${MUSETALK_LAUNCHER_DRY_RUN:-0}"; then
    log "DRY RUN: exec bash $legacy ${args[*]}"
    exit 0
  fi
  exec bash "$legacy" "${args[@]}"
}

case "$RECIPE" in
  legacy_int8)
    dispatch_legacy
    ;;
  fast|fast300|r5)
    ;;
  *)
    die "Unsupported MUSETALK_RECIPE=$RECIPE (source=$RECIPE_SOURCE); expected fast, fast300, r5 or legacy_int8"
    ;;
esac

# ------------------------------------------------------------------ layer 1 hygiene + layer 2
# Old-launcher parity: drop stale caller values from earlier experiments. An overrides file
# may still set any of them deliberately (it is loaded after this).
for stale_key in PYTORCH_CUDA_ALLOC_CONF MUSETALK_CPU_TUNING MUSETALK_CPU_THREADS \
  MUSETALK_CPU_INTEROP_THREADS MUSETALK_CPU_CV2_THREADS MUSETALK_CPU_NUMA_NODE \
  MUSETALK_CPU_AFFINITY HLS_CHUNK_ENCODER_TUNE HLS_CHUNK_ENCODER_QP; do
  if [[ -n "${!stale_key+x}" ]]; then
    log "unset caller $stale_key (old-launcher parity; put it in an overrides file to keep it)"
    unset "$stale_key"
  fi
done
unset stale_key

mt_env_record_caller
mt_env_load_overrides "$REPO_ROOT"
if (( ${#MT_ENV_FILES_USED[@]} == 0 )); then
  log "no overrides file present (looked for: ${MT_ENV_OVERRIDES_FILES[*]:-none})"
fi

# ------------------------------------------------------------------ layer 3: resolver
knob_keys_from_source() {
  # $1 = caller|overrides : comma list of MuseTalk-ish keys whose value came from that layer
  local want="$1" key src out=""
  for key in "${!MT_ENV_SOURCE[@]}"; do
    src="${MT_ENV_SOURCE[$key]}"
    [[ "$key" =~ ^(MUSETALK_|HLS_|WEBRTC_|AVATAR_|GPU_|PROFILE$|PYTHON|KOKORO_|HF_|TRANSFORMERS_|TORCHINDUCTOR_|MALLOC_|SERVER_) ]] || continue
    case "$want:$src" in
      caller:caller|overrides:overrides:*)
        out+="${out:+,}$key"
        ;;
    esac
  done
  printf '%s' "$out"
}

run_resolver() {
  local resolver="${MUSETALK_RESOLVER:-$REPO_ROOT/scripts/musetalk_host_profile.py}"
  local resolver_py="${MUSETALK_RESOLVER_PYTHON:-$VENV_PY}"
  local out_env out_json tmp_env tmp_json rc=0 used_files=""
  [[ -f "$resolver" ]] || die "Resolver not found: $resolver (component A of the startup rework)"

  if (( PRINT_ENV )); then
    LAUNCH_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/musetalk_print_env.XXXXXX")"
    out_env="$LAUNCH_TMP_DIR/musetalk_resolved.env"
    out_json="$LAUNCH_TMP_DIR/musetalk_resolved.json"
  elif (( VALIDATE_ONLY )); then
    out_env="${RESOLVED_ENV%.env}.validate.env"
    out_json="${RESOLVED_JSON%.json}.validate.json"
  else
    out_env="$RESOLVED_ENV"
    out_json="$RESOLVED_JSON"
  fi
  mkdir -p "$(dirname "$out_env")" "$(dirname "$out_json")"
  tmp_env="$out_env.tmp.$$"
  tmp_json="$out_json.tmp.$$"
  rm -f "$tmp_env" "$tmp_json"
  if (( ${#MT_ENV_FILES_USED[@]} > 0 )); then
    used_files="$(IFS=':'; printf '%s' "${MT_ENV_FILES_USED[*]}")"
  fi

  log "resolving recipe=$RECIPE with $resolver"
  (
    cd "$REPO_ROOT"
    MUSETALK_ENV_CALLER_KEYS="$(knob_keys_from_source caller)" \
    MUSETALK_ENV_OVERRIDE_KEYS="$(knob_keys_from_source overrides)" \
    MUSETALK_ENV_OVERRIDES_USED="$used_files" \
      "$resolver_py" -B "$resolver" resolve \
        --repo-root "$REPO_ROOT" \
        --venv "$VENV_PATH" \
        --out "$tmp_env" \
        --report "$tmp_json" \
        --recipe "$RECIPE"
  ) || rc=$?

  if (( rc != 0 )); then
    if [[ -f "$tmp_json" ]]; then
      "$VENV_PY" -I -B - "$tmp_json" <<'PY' >&2 || true
import json, sys
try:
    report = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for item in report.get("errors") or []:
    print(f"[resolver] error: {item}")
PY
    fi
    rm -f "$tmp_env" "$tmp_json"
    if (( rc == 2 )); then
      die "Resolver refused this host/config for recipe=$RECIPE (exit 2, reasons above). Fix the cause, or roll back with MUSETALK_RECIPE=legacy_int8."
    fi
    die "Resolver failed (exit $rc) for recipe=$RECIPE"
  fi
  [[ -s "$tmp_env" ]] || die "Resolver exited 0 but wrote no env file ($tmp_env)"
  mv -f "$tmp_env" "$out_env"
  if [[ -f "$tmp_json" ]]; then
    mv -f "$tmp_json" "$out_json"
  else
    warn "Resolver wrote no JSON report ($out_json)"
  fi
  RESOLVED_ENV_USED="$out_env"
  RESOLVED_JSON_USED="$out_json"
}

run_resolver
mt_env_load_file "$RESOLVED_ENV_USED" "resolved"
log "resolved env $RESOLVED_ENV_USED: exported=$MT_ENV_LAST_LOADED kept_higher_layer=$MT_ENV_LAST_KEPT invalid=$MT_ENV_LAST_INVALID"
if (( MT_ENV_LAST_INVALID > 0 )); then
  die "Resolved env $RESOLVED_ENV_USED has $MT_ENV_LAST_INVALID unparseable line(s)"
fi
if [[ -z "${MUSETALK_RECIPE+x}" ]]; then
  export MUSETALK_RECIPE="$RECIPE"
  MT_ENV_SOURCE[MUSETALK_RECIPE]="launcher:recipe"
fi

# ------------------------------------------------------------------ preflights
set_launcher_value() {
  # set_launcher_value KEY VALUE WHY: the launcher itself changes a value (logged + tracked)
  export "$1=$2"
  MT_ENV_SOURCE[$1]="launcher:$3"
}

preflight_vp8() {
  local mode="${WEBRTC_VP8_ENCODER:-pyav}" dir out rc=0 started
  mode="${mode,,}"
  mode="${mode//[[:space:]]/}"
  case "$mode" in
    pyav|"")
      return 0
      ;;
    native)
      ;;
    *)
      die "Unsupported WEBRTC_VP8_ENCODER=$mode (expected pyav or native)"
      ;;
  esac
  dir="${WEBRTC_NATIVE_VP8_DIR:-$REPO_ROOT/.runtime/native_vp8}"
  started="$(date +%s)"
  out="$(
    cd "$REPO_ROOT"
    WEBRTC_VP8_ENCODER=native WEBRTC_NATIVE_VP8_DIR="$dir" CUDA_VISIBLE_DEVICES= \
      timeout "${MUSETALK_VP8_PREFLIGHT_TIMEOUT_SECONDS:-60}" \
      "$VENV_PY" -B -c "from scripts import webrtc_native_vp8 as n; n.configure_vp8_encoder('preflight')" 2>&1
  )" || rc=$?
  if (( rc == 0 )); then
    log "native VP8 preflight passed in $(( $(date +%s) - started ))s (dir=$dir)"
    return 0
  fi
  printf '%s\n' "$out" | tail -n 15 >&2
  if mt_env_is_true "${MUSETALK_VP8_FALLBACK:-0}"; then
    warn "native VP8 preflight failed (exit $rc); MUSETALK_VP8_FALLBACK=1 -> WEBRTC_VP8_ENCODER=pyav"
    set_launcher_value WEBRTC_VP8_ENCODER pyav vp8_fallback
    return 0
  fi
  die "native VP8 preflight failed (exit $rc, dir=$dir). Install it (python scripts/install_native_vp8.py && --verify), set WEBRTC_VP8_ENCODER=pyav, or MUSETALK_VP8_FALLBACK=1."
}

repo_path() {
  if [[ "$1" == /* ]]; then
    printf '%s' "$1"
  else
    printf '%s/%s' "$REPO_ROOT" "$1"
  fi
}

csv_equal() {
  # compare two comma lists ignoring spaces
  local a="${1//[[:space:]]/}" b="${2//[[:space:]]/}"
  [[ "$a" == "$b" ]]
}

csv_max() {
  local list="${1//[[:space:]]/}" item max=0
  local -a items=()
  IFS=',' read -r -a items <<< "$list" || true
  for item in "${items[@]}"; do
    [[ "$item" =~ ^[0-9]+$ ]] || continue
    (( item > max )) && max="$item"
  done
  printf '%s' "$max"
}

preflight_levers() {
  local value fallback_on=0 manifest dir batch buckets
  if mt_env_is_true "${MUSETALK_TRT_FALLBACK:-1}"; then
    fallback_on=1
  fi

  # Values the server rejects at import/startup: fail here, before a 20-60 s model load.
  if [[ -n "${WEBRTC_H264_IMPL:-}" ]]; then
    value="${WEBRTC_H264_IMPL,,}"
    case "${value//[[:space:]]/}" in
      aiortc|x264tuned|nvenc) ;;
      *) die "Unsupported WEBRTC_H264_IMPL=$WEBRTC_H264_IMPL (expected aiortc, x264tuned or nvenc)" ;;
    esac
  fi
  if [[ -n "${MUSETALK_TRT_UNET_CUDAGRAPHS:-}" ]]; then
    value="${MUSETALK_TRT_UNET_CUDAGRAPHS,,}"
    case "${value//[[:space:]]/}" in
      0|off|false|no|none|1|on|true|yes|manual|runtime) ;;
      *) die "Invalid MUSETALK_TRT_UNET_CUDAGRAPHS=$MUSETALK_TRT_UNET_CUDAGRAPHS (expected manual, runtime or 0)" ;;
    esac
  fi
  if [[ -n "${HLS_SCHEDULER_POLICY:-}" ]]; then
    value="${HLS_SCHEDULER_POLICY,,}"
    case "${value//[[:space:]]/}" in
      roundrobin|edf) ;;
      *) die "Invalid HLS_SCHEDULER_POLICY=$HLS_SCHEDULER_POLICY (expected roundrobin or edf)" ;;
    esac
  fi
  if [[ "${WEBRTC_VP8_ENCODER:-pyav}" == native && -n "${WEBRTC_NATIVE_VP8_THREADS:-}" ]]; then
    if ! [[ "$WEBRTC_NATIVE_VP8_THREADS" =~ ^[0-9]+$ ]] || (( WEBRTC_NATIVE_VP8_THREADS < 1 || WEBRTC_NATIVE_VP8_THREADS > 16 )); then
      die "WEBRTC_NATIVE_VP8_THREADS must be an integer 1..16 (got $WEBRTC_NATIVE_VP8_THREADS)"
    fi
  fi

  # Stagewise FP16 UNet: no build-on-first-use and no fallback to the .ts.
  value="${MUSETALK_UNET_BACKEND:-}"
  case "${value,,}" in
    trt_stagewise|tensorrt_stagewise)
      batch="${MUSETALK_UNET_STAGEWISE_BATCH:-16}"
      dir="$(repo_path "${MUSETALK_UNET_STAGEWISE_CACHE_DIR:-models/tensorrt_unet_stagewise_sm89}")"
      manifest="$dir/bs$batch/manifest.json"
      if [[ ! -f "$manifest" ]] || ! grep -Eq '"complete"[[:space:]]*:[[:space:]]*true' "$manifest"; then
        if (( fallback_on )); then
          warn "MUSETALK_UNET_BACKEND=trt_stagewise but $manifest is missing/incomplete; with MUSETALK_TRT_FALLBACK=1 the server will SILENTLY serve the eager UNet"
        else
          die "MUSETALK_UNET_BACKEND=trt_stagewise but $manifest is missing or incomplete (build/adopt it: scripts/unet_engine_store.py ensure --kind unet_stagewise --batch $batch)"
        fi
      fi
      buckets="${HLS_SCHEDULER_FIXED_BATCH_SIZES:-}"
      if [[ -n "$buckets" ]] && ! csv_equal "$buckets" "$batch"; then
        warn "stagewise UNet engine batch $batch != HLS_SCHEDULER_FIXED_BATCH_SIZES=$buckets (smaller batches are padded, larger ones split)"
      fi
      ;;
  esac

  # TAESD TRT: STRICT=0 (code default) falls back to compiled TAESD with only a warning.
  value="${MUSETALK_TAESD_BACKEND:-}"
  case "${value,,}" in
    trt|tensorrt)
      dir="$(repo_path "${MUSETALK_TAESD_TRT_DIR:-models/taesd/trt}")"
      if ! compgen -G "$dir/taesd_trt_*.json" >/dev/null 2>&1; then
        if ! mt_env_is_true "${MUSETALK_TAESD_TRT_BUILD:-1}"; then
          if mt_env_is_true "${MUSETALK_TAESD_TRT_STRICT:-0}"; then
            die "MUSETALK_TAESD_BACKEND=trt with BUILD=0 and STRICT=1, but no engine in $dir (scripts/unet_engine_store.py ensure --kind taesd_trt)"
          fi
          warn "MUSETALK_TAESD_BACKEND=trt with BUILD=0 but no engine in $dir: the server will fall back to compiled TAESD"
        else
          warn "no TAESD TRT engine in $dir yet: the server builds one on first use (~15-60 s inside startup)"
        fi
      fi
      ;;
  esac

  if mt_env_is_true "${MUSETALK_FREE_EAGER_UNET:-0}" && mt_env_is_true "${MUSETALK_UNET_CALIBRATION_CAPTURE:-0}"; then
    warn "MUSETALK_FREE_EAGER_UNET=1 with MUSETALK_UNET_CALIBRATION_CAPTURE=1: capture needs the eager UNet; keep FREE_EAGER_UNET=0"
  fi

  # Bucket coupling (the resolver emits a coupled set; a caller/overrides value can break it).
  buckets="${HLS_SCHEDULER_FIXED_BATCH_SIZES:-}"
  if [[ -n "$buckets" ]]; then
    if [[ -n "${MUSETALK_TAESD_WARMUP_BATCHES:-}" ]] && ! csv_equal "$buckets" "$MUSETALK_TAESD_WARMUP_BATCHES"; then
      warn "MUSETALK_TAESD_WARMUP_BATCHES=$MUSETALK_TAESD_WARMUP_BATCHES != HLS_SCHEDULER_FIXED_BATCH_SIZES=$buckets (unwarmed buckets compile on live traffic)"
    fi
    if [[ -n "${MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES:-}" ]] && ! csv_equal "$buckets" "$MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES"; then
      warn "MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=$MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES != HLS_SCHEDULER_FIXED_BATCH_SIZES=$buckets (WebRTC batch snapping differs)"
    fi
    if [[ -n "${HLS_SCHEDULER_MAX_BATCH:-}" && "$(csv_max "$buckets")" != "$HLS_SCHEDULER_MAX_BATCH" ]]; then
      warn "HLS_SCHEDULER_MAX_BATCH=$HLS_SCHEDULER_MAX_BATCH != max(HLS_SCHEDULER_FIXED_BATCH_SIZES=$buckets)"
    fi
  fi
  if mt_env_is_true "${MUSETALK_TRT_FALLBACK:-1}"; then
    warn "MUSETALK_TRT_FALLBACK is on: a failed TAESD/TRT load silently serves a slower backend (recipe verification after /health catches it)"
  fi
}

compute_expectations() {
  local vae="${MUSETALK_VAE_BACKEND:-}" taesd="${MUSETALK_TAESD_BACKEND:-}" unet="${MUSETALK_UNET_BACKEND:-}"
  vae="${vae,,}"; taesd="${taesd,,}"; unet="${unet,,}"
  case "$vae" in
    taesd|tiny|tiny_vae)
      case "$taesd" in
        trt|tensorrt) EXPECT_VAE="taesd_trt" ;;
        *) EXPECT_VAE="taesd" ;;
      esac
      ;;
    trt_stagewise|tensorrt_stagewise)
      EXPECT_VAE="trt_stagewise"
      ;;
    "")
      EXPECT_VAE="pytorch"
      ;;
    *)
      EXPECT_VAE="any"
      ;;
  esac
  case "$unet" in
    trt_stagewise|tensorrt_stagewise) EXPECT_UNET="trt_stagewise" ;;
    trt|tensorrt) EXPECT_UNET="trt" ;;
    *)
      if mt_env_is_true "${MUSETALK_TRT_UNET_ENABLED:-0}"; then
        EXPECT_UNET="trt"
      elif [[ -z "$unet" || "$unet" == eager || "$unet" == pytorch ]]; then
        EXPECT_UNET="eager"
      else
        EXPECT_UNET="any"
      fi
      ;;
  esac
}

preflight_vp8
preflight_levers
compute_expectations

# ------------------------------------------------------------------ reporting
managed_keys() {
  {
    mt_env_file_keys "$RESOLVED_ENV_USED"
    printf '%s\n' "${MT_PASSTHROUGH_LEVERS[@]}" "${MT_LAUNCH_CONTROL_KNOBS[@]}"
    local recipe_file
    for recipe_file in "$RECIPE_FILES_DIR"/*.env; do
      [[ -f "$recipe_file" ]] || continue
      # "#KEY=VALUE" is a switched-off lever, "# ..." is prose (configs/recipes/fast300.env format)
      sed -nE 's/^#?([A-Z][A-Z0-9_]*)=.*/\1/p' "$recipe_file"
    done
    local key
    for key in "${!MT_ENV_SOURCE[@]}"; do
      case "${MT_ENV_SOURCE[$key]}" in
        overrides:*|launcher:*) printf '%s\n' "$key" ;;
      esac
    done
  } | awk 'NF && !seen[$0]++' | LC_ALL=C sort
}

key_source() {
  local key="$1"
  if [[ -n "${!key+x}" ]]; then
    printf '%s' "${MT_ENV_SOURCE[$key]:-caller}"
  else
    printf 'unset'
  fi
}

print_env() {
  local key
  printf '# MuseTalk effective env (recipe=%s, source=%s)\n' "$RECIPE" "$RECIPE_SOURCE"
  printf '# layers: caller > overrides(%s) > resolved(%s) > code default\n' \
    "${MT_ENV_FILES_USED[*]:-none}" "$RESOLVED_ENV_USED"
  printf '# expect: vae=%s unet=%s\n' "$EXPECT_VAE" "$EXPECT_UNET"
  while IFS= read -r key; do
    [[ -n "$key" ]] || continue
    if [[ -n "${!key+x}" ]]; then
      printf '%s=%s  # %s\n' "$key" "$(mt_env_display "$key" "${!key}")" "$(key_source "$key")"
    else
      printf '#%s=  # unset (code default)\n' "$key"
    fi
  done < <(managed_keys)
}

print_summary() {
  local key levers="" src
  log "recipe=$RECIPE (source=$RECIPE_SOURCE) host=$HOST port=$PORT venv=$VENV_PATH"
  log "vae=${MUSETALK_VAE_BACKEND:-pytorch} taesd_backend=${MUSETALK_TAESD_BACKEND:-compiled} taesd_compile=${MUSETALK_TAESD_COMPILE:-1} warmup=${MUSETALK_TAESD_WARMUP_BATCHES:-8}"
  case "${MUSETALK_UNET_BACKEND:-}" in
    trt_stagewise|tensorrt_stagewise)
      log "unet=${MUSETALK_UNET_BACKEND} engine=$(repo_path "${MUSETALK_UNET_STAGEWISE_CACHE_DIR:-models/tensorrt_unet_stagewise_sm89}")/bs${MUSETALK_UNET_STAGEWISE_BATCH:-16} cudagraph=${MUSETALK_UNET_STAGEWISE_CUDAGRAPH:-1} fallback=${MUSETALK_TRT_FALLBACK:-1(code)}"
      ;;
    *)
      log "unet=${MUSETALK_UNET_BACKEND:-eager} trt_unet_enabled=${MUSETALK_TRT_UNET_ENABLED:-0} engine=${MUSETALK_TRT_UNET_PATHS:-none} cudagraphs=${MUSETALK_TRT_UNET_CUDAGRAPHS:-0} fallback=${MUSETALK_TRT_FALLBACK:-1(code)}"
      ;;
  esac
  log "buckets=${HLS_SCHEDULER_FIXED_BATCH_SIZES:-auto} max_batch=${HLS_SCHEDULER_MAX_BATCH:-auto} slice=${HLS_SCHEDULER_STARTUP_SLICE_SIZE:-auto} stagewise_warmup=${MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES:-auto}"
  log "workers prep/compose/encode=${HLS_PREP_WORKERS:-2}/${HLS_COMPOSE_WORKERS:-2}/${HLS_ENCODE_WORKERS:-2} avatar_load=${MUSETALK_AVATAR_LOAD_WORKERS:-auto} pending=${HLS_MAX_PENDING_JOBS:-16} cache_mb=${AVATAR_CACHE_MAX_MEMORY_MB:-auto} gpu_total_gb=${GPU_TOTAL_MEMORY_GB:-auto}"
  log "vp8=${WEBRTC_VP8_ENCODER:-pyav} h264_impl=${WEBRTC_H264_IMPL:-aiortc} relay_policy=${WEBRTC_ICE_TRANSPORT_POLICY:-all}"
  log "overrides=${MT_ENV_FILES_USED[*]:-none} resolved=$RESOLVED_ENV_USED report=$RESOLVED_JSON_USED"
  log "expect after /health: vae=$EXPECT_VAE unet=$EXPECT_UNET"
  for key in "${MT_PASSTHROUGH_LEVERS[@]}"; do
    [[ -n "${!key+x}" ]] || continue
    src="$(key_source "$key")"
    levers+=" $key=$(mt_env_display "$key" "${!key}")($src)"
  done
  log "levers:${levers:- none set (all code defaults)}"
}

write_launch_state() {
  local state="$1" key tmp
  mkdir -p "$(dirname "$state")"
  tmp="$state.tmp.$$"
  {
    while IFS= read -r key; do
      [[ -n "$key" && -n "${!key+x}" ]] || continue
      if mt_env_is_secret_key "$key"; then
        printf '%s\0%s\0%s\0' "$key" "<redacted>" "$(key_source "$key")"
      else
        printf '%s\0%s\0%s\0' "$key" "${!key}" "$(key_source "$key")"
      fi
    done < <(managed_keys)
  } | MT_STATE_RECIPE="$RECIPE" MT_STATE_RECIPE_SOURCE="$RECIPE_SOURCE" MT_STATE_HOST="$HOST" \
      MT_STATE_PORT="$PORT" MT_STATE_PID="$$" MT_STATE_REPO="$REPO_ROOT" MT_STATE_VENV="$VENV_PATH" \
      MT_STATE_RESOLVED_ENV="$RESOLVED_ENV_USED" MT_STATE_RESOLVED_JSON="$RESOLVED_JSON_USED" \
      MT_STATE_OVERRIDES="$(IFS=':'; printf '%s' "${MT_ENV_FILES_USED[*]:-}")" \
      MT_STATE_EXPECT_VAE="$EXPECT_VAE" MT_STATE_EXPECT_UNET="$EXPECT_UNET" \
      MT_STATE_PROFILE_ARG="$LAUNCH_PROFILE_ARG" \
      MT_STATE_WARNINGS="$(printf '%s\n' "${MT_ENV_WARNINGS[@]:-}" "${LAUNCH_WARNINGS[@]:-}")" \
      "$VENV_PY" -I -B -c '
import json, os, sys, time
raw = sys.stdin.buffer.read().split(b"\0")
env = {}
for i in range(0, len(raw) - 2, 3):
    key, value, source = (part.decode("utf-8", "replace") for part in raw[i:i + 3])
    env[key] = {"value": value, "source": source}
g = os.environ.get
state = {
    "schema": "musetalk_launch_v1",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "created_ts": time.time(),
    "launcher_pid": int(g("MT_STATE_PID", "0")),
    "host": g("MT_STATE_HOST"),
    "port": g("MT_STATE_PORT"),
    "recipe": g("MT_STATE_RECIPE"),
    "recipe_source": g("MT_STATE_RECIPE_SOURCE"),
    "profile_arg": g("MT_STATE_PROFILE_ARG") or None,
    "repo_root": g("MT_STATE_REPO"),
    "venv": g("MT_STATE_VENV"),
    "resolved_env": g("MT_STATE_RESOLVED_ENV"),
    "resolved_report": g("MT_STATE_RESOLVED_JSON"),
    "overrides_files": [p for p in (g("MT_STATE_OVERRIDES") or "").split(":") if p],
    "expect": {"vae": g("MT_STATE_EXPECT_VAE"), "unet": g("MT_STATE_EXPECT_UNET")},
    "warnings": [w for w in (g("MT_STATE_WARNINGS") or "").splitlines() if w.strip()],
    "env": env,
}
json.dump(state, sys.stdout, indent=1, sort_keys=True)
sys.stdout.write("\n")
' > "$tmp"
  mv -f "$tmp" "$state"
}

print_summary

if (( PRINT_ENV )); then
  print_env
  exit 0
fi

if (( VALIDATE_ONLY )); then
  log "Validation-only checks passed in $(( $(date +%s) - LAUNCH_START_TS ))s; API server was not started (resolver output: $RESOLVED_ENV_USED)"
  exit 0
fi

write_launch_state "$LAUNCH_STATE"
log "launch state: $LAUNCH_STATE"

if mt_env_is_true "${MUSETALK_LAUNCHER_DRY_RUN:-0}"; then
  log "DRY RUN: cd $REPO_ROOT && exec $VENV_PY api_server.py --host $HOST --port $PORT"
  exit 0
fi

log "exec api_server.py (pre-launch took $(( $(date +%s) - LAUNCH_START_TS ))s)"
cleanup_tmp
trap - EXIT
cd "$REPO_ROOT"
exec "$VENV_PY" api_server.py --host "$HOST" --port "$PORT"
