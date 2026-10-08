#!/usr/bin/env bash
# install_musetalk.sh - canonical MuseTalk installer (fast recipe on any NVIDIA GPU).
#
# Builds (or repairs in place) the single server venv from requirements/server.in plus the
# pinned requirements/constraints-<matrix>.txt, fetches weights (incl. pinned TAESD and the
# optional Kokoro cache), provisions native VP8, optionally a dedicated chin-tracker venv, runs a
# CPU import smoke with CUDA hidden, a GPU self-test when a GPU is visible, and stamps
# <repo>/.runtime/install_state.json.  `--check` is read-only (exit 0 ok, 10 clean install
# needed, 11 repairable in place).  scripts/setup_musetalk.sh is a shim that execs this file.
#
# Never requires CUDA to install (GPU-less image builds work); never deletes a venv unless
# --clean is given.  The installer's own logic never imports torch: all venv inspection goes
# through scripts/musetalk_install_state.py (stdlib only, dist-info + find_spec).
set -Eeuo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=lib/step_logging.sh
source "$SCRIPT_DIR/lib/step_logging.sh"

EXIT_NEEDS_CLEAN=10
EXIT_REPAIRABLE=11

MODE="install"                      # install | check | plan
VENV_PATH="${VENV_PATH:-}"
MATRIX="${MUSETALK_INSTALL_MATRIX:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
CLEAN=0
SKIP_APT=0
SKIP_WEIGHTS=0
KOKORO=""                           # "" = stamp/default (on); 1/0 = explicit
NATIVE_VP8=""                       # "" = stamp/auto; 1/0 = explicit
AVATAR_PREP=""
LEGACY_INT8=""
CHIN_TOOLS=""
CHIN_VENV="${MUSETALK_CHIN_TOOLS_VENV:-}"
CHIN_VENV_EXPLICIT=0
[[ -n "$CHIN_VENV" ]] && CHIN_VENV_EXPLICIT=1
SELFTEST=1
SELFTEST_UNET=0
CHECK_IMPORTS=0
CHECK_REPORT=""
CHECK_JSON=0
NATIVE_VP8_WHEEL="${MUSETALK_NATIVE_VP8_WHEEL:-}"
NATIVE_VP8_DIR="${WEBRTC_NATIVE_VP8_DIR:-}"
STATE_FILE="${MUSETALK_INSTALL_STATE_FILE:-}"
SELFTEST_OUT="${MUSETALK_GPU_SELFTEST_FILE:-}"
PIP_VERSION="${MUSETALK_PIP_VERSION:-26.2.1}"
PYTORCH_INDEX_BASE="${MUSETALK_PYTORCH_INDEX_BASE:-https://download.pytorch.org/whl}"
NVIDIA_INDEX_URL="${MUSETALK_NVIDIA_INDEX_URL:-https://pypi.nvidia.com}"
WHEELHOUSE="${MUSETALK_WHEELHOUSE:-}"
MIN_DISK_GB_FRESH="${MUSETALK_INSTALL_MIN_DISK_GB:-20}"
MIN_DISK_GB_REPAIR="${MUSETALK_INSTALL_MIN_DISK_GB_REPAIR:-4}"
SELFTEST_TIMEOUT_S="${MUSETALK_SELFTEST_TIMEOUT_S:-1200}"
MMCV_VERSION="${MMCV_VERSION:-2.1.0}"
MMCV_BUILD_MAX_JOBS="${MMCV_BUILD_MAX_JOBS:-8}"

log() { step_log_emit INFO "$*"; }
warn() { step_log_emit WARN "$*"; }
die() {
  local code=1
  if [[ "${1:-}" =~ ^[0-9]+$ ]]; then code="$1"; shift; fi
  step_log_emit ERROR "$*"
  exit "$code"
}

on_err() {
  local status="$1" line="$2" cmd="$3"
  case "$cmd" in return*) return 0 ;; esac  # a function reporting its status is not a new failure
  step_log_emit ERROR "command failed (exit $status) at line $line: $cmd"
}
trap 'on_err "$?" "$LINENO" "$BASH_COMMAND"' ERR

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options]

Install or repair the MuseTalk server venv (default recipe: fast = compiled TAESD + TRT/eager UNet).

Modes:
  (default)               install / repair in place (idempotent)
  --check                 read-only verification: exit 0 ok, $EXIT_NEEDS_CLEAN clean install needed
                          (venv missing, wrong Python, wrong CUDA matrix), $EXIT_REPAIRABLE repairable in place
                          (missing packages/pins/weights/optional groups). A pre-existing venv without a
                          stamp that passes every check is OK (warning).
  --check-imports         with --check: also run the CPU import smoke (imports torch; CUDA hidden)
  --report FILE           with --check: write the JSON report to FILE
  --json                  with --check: print the JSON report to stdout
  --plan                  print the resolved plan (matrix, groups, pip command) and exit; no changes

Options:
  --venv PATH             server venv (default: \$WORKSPACE/.venvs/musetalk_trt_stagewise)
  --repo-root PATH        MuseTalk checkout (default: $REPO_ROOT)
  --matrix M              auto | cu121 | cu128 (auto: cu128 iff GPU compute capability >= 10.0;
                          no GPU -> cu121, or the installed matrix when a venv exists)
  --python BIN            base interpreter (default: $PYTHON_BIN; must be CPython 3.10)
  --clean                 delete and recreate the venv (never done implicitly)
  --skip-apt              skip apt packages (apt only runs as root anyway)
  --skip-weights          skip download_weights.sh
  --with-kokoro | --without-kokoro          local Kokoro TTS + model cache (default: with)
  --with-native-vp8 | --without-native-vp8  pinned native VP8 encoder files (default: auto =
                          with on x86_64 + CPython 3.10; failures are warnings unless explicit)
  --native-vp8-wheel PATH use this pinned aiortc 1.11.0 wheel (offline native VP8 install)
  --with-avatar-prep      mmcv/mmdet/mmpose for /avatars/prepare (cu121 only)
  --with-legacy-int8      nvidia-modelopt stack for MUSETALK_RECIPE=legacy_int8 (cu121 only)
  --with-chin-tools       dedicated MediaPipe venv (default \$WORKSPACE/.venvs/musetalk_chin_tools),
                          exported as MUSETALK_CHIN_TRACKER_PYTHON through the install stamp
  --chin-venv PATH        chin-tools venv location
  --no-selftest           skip the GPU self-test (scripts/musetalk_selftest.py)
  --selftest-unet         self-test also times the eager UNet at bs8 (+~4 GB host RAM)
  --selftest-out FILE     self-test JSON (default: <repo>/.runtime/gpu_selftest.json)
  --state-file FILE       install stamp (default: <repo>/.runtime/install_state.json)
  -h, --help              this help

Legacy aliases: --venv-path (= --venv), --python-bin (= --python), --full-stack / --install-avatar-prep-deps
(= --with-avatar-prep), --install-modelopt (= --with-legacy-int8), --skip-modelopt, --artifact-dir (ignored).

Environment: PIP_CACHE_DIR (respected), MUSETALK_WHEELHOUSE (extra --find-links dir),
  MUSETALK_PYTORCH_INDEX_BASE, MUSETALK_NVIDIA_INDEX_URL, MUSETALK_PIP_VERSION ($PIP_VERSION),
  MUSETALK_INSTALL_MIN_DISK_GB ($MIN_DISK_GB_FRESH, fresh venv), HF_MAX_WORKERS, HF_XET_HIGH_PERFORMANCE.
  DOWNLOAD_SYNCNET_WEIGHTS=1 opts into the training-only checkpoint (default 0).
  This does not install or validate a training environment; /avatars/prepare does not require it.
  PYTORCH_INDEX_URL from the base image is deliberately ignored (the matrix picks the index).

Examples:
  $SCRIPT_NAME                               # install or repair the default venv
  $SCRIPT_NAME --check                       # boot-time verification (read-only, <2 s)
  $SCRIPT_NAME --clean --matrix cu128        # Blackwell clean install
  $SCRIPT_NAME --with-avatar-prep --with-chin-tools
EOF
}

# ------------------------------------------------------------------------------- arguments
require_value() { [[ $# -ge 2 && -n "$2" ]] || die 2 "$1 requires a value"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv|--venv-path) require_value "$@"; VENV_PATH="$2"; shift 2 ;;
    --repo-root) require_value "$@"; REPO_ROOT="$2"; shift 2 ;;
    --matrix) require_value "$@"; MATRIX="$2"; shift 2 ;;
    --python|--python-bin) require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    --skip-apt) SKIP_APT=1; shift ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --with-kokoro) KOKORO=1; shift ;;
    --without-kokoro) KOKORO=0; shift ;;
    --with-native-vp8) NATIVE_VP8=1; shift ;;
    --without-native-vp8) NATIVE_VP8=0; shift ;;
    --native-vp8-wheel) require_value "$@"; NATIVE_VP8_WHEEL="$2"; shift 2 ;;
    --with-avatar-prep) AVATAR_PREP=1; shift ;;
    --with-legacy-int8) LEGACY_INT8=1; shift ;;
    --with-chin-tools) CHIN_TOOLS=1; shift ;;
    --chin-venv) require_value "$@"; CHIN_VENV="$2"; CHIN_VENV_EXPLICIT=1; shift 2 ;;
    --no-selftest) SELFTEST=0; shift ;;
    --selftest-unet) SELFTEST_UNET=1; shift ;;
    --selftest-out) require_value "$@"; SELFTEST_OUT="$2"; shift 2 ;;
    --state-file) require_value "$@"; STATE_FILE="$2"; shift 2 ;;
    --check) MODE="check"; shift ;;
    --check-imports) CHECK_IMPORTS=1; shift ;;
    --report) require_value "$@"; CHECK_REPORT="$2"; shift 2 ;;
    --json) CHECK_JSON=1; shift ;;
    --plan|--dry-run) MODE="plan"; shift ;;
    # Legacy setup_trt_stagewise_server_env.sh flags (scripts/setup_musetalk.sh translates them too).
    --full-stack|--install-avatar-prep-deps) AVATAR_PREP=1; shift ;;
    --install-modelopt) LEGACY_INT8=1; shift ;;
    --skip-modelopt) LEGACY_INT8=0; shift ;;
    --artifact-dir) require_value "$@"; step_log_emit INFO "Ignoring legacy --artifact-dir $2 (engines live in the engine store)"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die 2 "Unknown option: $1" ;;
  esac
done

case "$MATRIX" in auto|cu121|cu128) ;; *) die 2 "--matrix must be auto, cu121 or cu128 (got: $MATRIX)" ;; esac
[[ -d "$REPO_ROOT" ]] || die 2 "Repo root not found: $REPO_ROOT"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"

WORKSPACE_ROOT="${WORKSPACE:-}"
if [[ -z "$WORKSPACE_ROOT" ]]; then
  if [[ "$REPO_ROOT" == /workspace/* || "$REPO_ROOT" == "/workspace" ]]; then
    WORKSPACE_ROOT="/workspace"
  else
    WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
  fi
fi
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/.venvs/musetalk_trt_stagewise}"
CHIN_VENV="${CHIN_VENV:-$WORKSPACE_ROOT/.venvs/musetalk_chin_tools}"
STATE_FILE="${STATE_FILE:-$REPO_ROOT/.runtime/install_state.json}"
SELFTEST_OUT="${SELFTEST_OUT:-$REPO_ROOT/.runtime/gpu_selftest.json}"
NATIVE_VP8_DIR="${NATIVE_VP8_DIR:-$REPO_ROOT/.runtime/native_vp8}"
STEP_LOG_ROOT="${STEP_LOG_ROOT:-$REPO_ROOT/logs/setup}"
abs_path() { if [[ -z "$1" || "$1" == /* ]]; then printf '%s' "$1"; else printf '%s/%s' "$PWD" "$1"; fi; }
VENV_PATH="$(abs_path "$VENV_PATH")"
CHIN_VENV="$(abs_path "$CHIN_VENV")"
STATE_FILE="$(abs_path "$STATE_FILE")"
SELFTEST_OUT="$(abs_path "$SELFTEST_OUT")"
NATIVE_VP8_DIR="$(abs_path "$NATIVE_VP8_DIR")"
NATIVE_VP8_WHEEL="$(abs_path "$NATIVE_VP8_WHEEL")"
VENV_PY="$VENV_PATH/bin/python"
STATE_HELPER="$SCRIPT_DIR/musetalk_install_state.py"
REQ_DIR="$REPO_ROOT/requirements"

# Any CPython >= 3.8 runs the stdlib helper; prefer the requested base interpreter.
HOST_PY=""
for candidate in "$PYTHON_BIN" python3 /usr/bin/python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    HOST_PY="$(command -v "$candidate")"
    break
  fi
done

state_helper() {
  [[ -n "$HOST_PY" ]] || die "No python >= 3.8 found to run $STATE_HELPER (install python3.10 first)"
  "$HOST_PY" -B "$STATE_HELPER" "$@"
}

# apt is defined early: a bare image may need it before any python exists.
apt_packages_step() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  if ! command -v python3.10 >/dev/null 2>&1 && ! apt-cache show python3.10 >/dev/null 2>&1; then
    log "python3.10 is not in the distro repositories; adding ppa:deadsnakes/ppa"
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -y
  fi
  apt-get install -y \
    python3.10 python3.10-venv python3.10-dev \
    coturn ffmpeg git curl build-essential ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
}

# ------------------------------------------------------------------------------- check mode
# The import smoke is shared by install (always) and --check --check-imports (opt-in).
# It imports the heavy stack in ONE CPU-only process (CUDA hidden) and writes a JSON summary with
# versions, PyAV encoder availability (H.264 levers) and torch's compiled arch list.
run_import_smoke() {
  local out_json="$1" kokoro="$2" legacy="$3" avatar="$4"
  CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 SMOKE_KOKORO="$kokoro" SMOKE_LEGACY="$legacy" SMOKE_AVATAR="$avatar" \
    "$VENV_PY" -B - "$out_json" <<'PY'
import ctypes, importlib, json, os, sys, time
out = {"schema": "musetalk_import_smoke_v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "modules": {}, "failed": [], "deferred": []}
try:
    ctypes.CDLL("libcuda.so.1")  # loads the driver library only; no cuInit, no GPU context
    driver = True
except OSError:
    driver = False
out["nvidia_driver_library"] = driver
required = ["torch", "torchvision", "diffusers", "transformers", "cv2", "numpy", "aiortc", "av", "fastapi",
            "uvicorn", "boto3", "librosa", "soundfile", "onnx", "numba", "multipart", "omegaconf", "ffmpeg"]
gpu_stack = ["tensorrt", "torch_tensorrt"]
# torch_tensorrt 2.5 queries the CUDA device while it imports (a CompilationSettings() default calls
# torch.cuda.current_device()), so with CUDA hidden it can only fail with "No CUDA GPUs are available", on every
# host. That case is deferred to the GPU self-test, which imports it with the GPU visible; a missing or broken
# package still raises ImportError here and fails.
needs_device = {"torch_tensorrt"}
if os.environ.get("SMOKE_KOKORO") == "1":
    required += ["kokoro", "misaki", "spacy", "en_core_web_sm"]
if os.environ.get("SMOKE_LEGACY") == "1":
    required += ["modelopt.torch.quantization"]
if os.environ.get("SMOKE_AVATAR") == "1":
    required += ["mmengine", "mmcv", "mmcv._ext", "mmcv.ops", "mmdet", "mmpose"]
for name in required + gpu_stack:
    started = time.time()
    try:
        module = importlib.import_module(name)
        out["modules"][name] = {"ok": True, "version": str(getattr(module, "__version__", "")),
                                "seconds": round(time.time() - started, 2)}
    except Exception as exc:  # record every failure, keep going
        out["modules"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
        if name in gpu_stack and not driver:
            out["deferred"].append(name)  # GPU-less image build: TensorRT needs the driver library
        elif name in needs_device and not isinstance(exc, ImportError) and "CUDA" in str(exc):
            out["deferred"].append(name)  # needs a visible GPU at import: checked by the GPU self-test
        else:
            out["failed"].append(name)
try:
    import torch
    out["torch_cuda"] = torch.version.cuda
    out["torch_arch_flags"] = torch._C._cuda_getArchFlags() if hasattr(torch._C, "_cuda_getArchFlags") else None
except Exception as exc:
    out["torch_arch_flags"] = None
    out["torch_error"] = str(exc)[:200]
try:
    import av
    out["encoders"] = {name: name in av.codecs_available
                       for name in ("libx264", "h264_nvenc", "libopenh264", "libvpx", "libvpx-vp9")}
except Exception as exc:
    out["encoders"] = {"error": str(exc)[:200]}
out["ok"] = not out["failed"]
with open(sys.argv[1], "w") as handle:
    json.dump(out, handle, indent=2)
for name, info in out["modules"].items():
    status = "ok" if info["ok"] else ("DEFERRED" if name in out["deferred"] else "FAIL")
    print(f"[import-smoke] {status:8s} {name} {info.get('version') or info.get('error', '')}")
print(f"[import-smoke] encoders={out.get('encoders')} torch_arch_flags={out.get('torch_arch_flags')}")
sys.exit(0 if out["ok"] else 1)
PY
}

check_mode() {
  local -a args=(check --repo-root "$REPO_ROOT" --venv "$VENV_PATH" --matrix "$MATRIX"
                 --native-vp8-dir "$NATIVE_VP8_DIR" --state-file "$STATE_FILE")
  # Explicit chin venv only; otherwise the helper uses the stamp's path, then the default.
  if (( CHIN_VENV_EXPLICIT )); then args+=(--chin-venv "$CHIN_VENV"); fi
  [[ -n "$KOKORO" ]] && args+=(--kokoro "$KOKORO")
  [[ -n "$NATIVE_VP8" ]] && args+=(--native-vp8 "$NATIVE_VP8")
  [[ -n "$AVATAR_PREP" ]] && args+=(--avatar-prep "$AVATAR_PREP")
  [[ -n "$LEGACY_INT8" ]] && args+=(--legacy-int8 "$LEGACY_INT8")
  [[ -n "$CHIN_TOOLS" ]] && args+=(--chin-tools "$CHIN_TOOLS")
  (( SKIP_WEIGHTS )) && args+=(--skip-weights)
  [[ -n "$CHECK_REPORT" ]] && args+=(--report "$CHECK_REPORT")
  (( CHECK_JSON )) && args+=(--json)
  local status=0
  state_helper "${args[@]}" || status=$?
  if (( status == EXIT_NEEDS_CLEAN )) || (( CHECK_IMPORTS == 0 )); then
    return "$status"
  fi
  local smoke_dir smoke_status=0
  smoke_dir="$(mktemp -d "${TMPDIR:-/tmp}/musetalk-check-XXXXXX")"
  run_import_smoke "$smoke_dir/smoke.json" "${KOKORO:-1}" "${LEGACY_INT8:-0}" "${AVATAR_PREP:-0}" >&2 || smoke_status=$?
  rm -rf "$smoke_dir"
  if (( smoke_status != 0 )); then
    step_log_emit WARN "CPU import smoke failed -> repairable in place (exit $EXIT_REPAIRABLE)"
    return "$EXIT_REPAIRABLE"
  fi
  return "$status"
}

if [[ "$MODE" == "check" ]]; then
  STEP_LOG_SCRIPT_NAME="$SCRIPT_NAME"
  trap - ERR  # exit 10/11 are verdicts, not errors
  set +e
  check_mode
  status=$?
  set -e
  exit "$status"
fi

# ------------------------------------------------------------------------------- plan
flag_on() { [[ "$1" == "1" ]]; }

stamp_group() {
  # Group value from an existing stamp for THIS venv ("" when unknown).
  [[ -f "$STATE_FILE" && -n "$HOST_PY" ]] || return 0
  "$HOST_PY" -B - "$STATE_FILE" "$VENV_PATH" "$1" <<'PY' 2>/dev/null || true
import json, os, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
if os.path.realpath(data.get("venv", "")) != os.path.realpath(sys.argv[2]):
    sys.exit(0)
value = (data.get("groups") or {}).get(sys.argv[3])
if isinstance(value, dict):
    value = value.get("enabled")
if value is not None:
    print("1" if value else "0")
PY
}

resolve_plan() {
  VENV_EXISTS=0
  VENV_TORCH_TAG=""
  if [[ -e "$VENV_PATH" ]]; then
    VENV_EXISTS=1
    VENV_TORCH_TAG="$(state_helper venv-facts --venv "$VENV_PATH" --field torch_cuda_tag 2>/dev/null || true)"
  fi
  local detected
  detected="$(state_helper detect-matrix --matrix "$MATRIX" --shell)" || die "GPU/matrix detection failed"
  # shlex-quoted KEY=VALUE lines from our own helper (DETECTED_MATRIX, _REASON, _GPU_VISIBLE, _GPU_DESC).
  eval "$detected"
  TARGET_MATRIX="$DETECTED_MATRIX"
  MATRIX_REASON="$DETECTED_REASON"
  GPU_VISIBLE="$DETECTED_GPU_VISIBLE"
  GPU_DESC="$DETECTED_GPU_DESC"
  if [[ "$MATRIX" == "auto" && "$GPU_VISIBLE" == "0" && ( "$VENV_TORCH_TAG" == "cu121" || "$VENV_TORCH_TAG" == "cu128" ) && $CLEAN -eq 0 ]]; then
    TARGET_MATRIX="$VENV_TORCH_TAG"
    MATRIX_REASON="no GPU visible: keeping the installed $VENV_TORCH_TAG matrix"
  fi

  local value
  if [[ -z "$KOKORO" ]]; then value="$(stamp_group kokoro)"; KOKORO="${value:-1}"; fi
  if [[ -z "$AVATAR_PREP" ]]; then value="$(stamp_group avatar_prep)"; AVATAR_PREP="${value:-0}"; fi
  if [[ -z "$LEGACY_INT8" ]]; then value="$(stamp_group legacy_int8)"; LEGACY_INT8="${value:-0}"; fi
  if [[ -z "$CHIN_TOOLS" ]]; then value="$(stamp_group chin_tools)"; CHIN_TOOLS="${value:-0}"; fi
  NATIVE_VP8_EXPLICIT=0
  if [[ -n "$NATIVE_VP8" ]]; then
    NATIVE_VP8_EXPLICIT=1
  else
    if [[ "$(uname -m)" == "x86_64" ]] && "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)' 2>/dev/null; then
      NATIVE_VP8=1
    else
      NATIVE_VP8=0
    fi
  fi

  if [[ "$TARGET_MATRIX" == "cu128" ]] && flag_on "$AVATAR_PREP"; then
    die 2 "--with-avatar-prep is cu121-only (the mmcv 2.1.0 wheel targets torch 2.5/cu121)"
  fi
  if [[ "$TARGET_MATRIX" == "cu128" ]] && flag_on "$LEGACY_INT8"; then
    die 2 "--with-legacy-int8 is cu121-only (nvidia-modelopt 0.23.2 validated with torch 2.5.1)"
  fi

  CONSTRAINTS="$REQ_DIR/constraints-$TARGET_MATRIX.txt"
  [[ -f "$CONSTRAINTS" ]] || die "Constraints file missing: $CONSTRAINTS"
  [[ -f "$REQ_DIR/server.in" ]] || die "requirements/server.in missing under $REPO_ROOT"
  PYTORCH_INDEX="$PYTORCH_INDEX_BASE/$TARGET_MATRIX"

  PIP_INSTALL_ARGS=(install --disable-pip-version-check -r "$REQ_DIR/server.in")
  if flag_on "$KOKORO"; then PIP_INSTALL_ARGS+=(-r "$REQ_DIR/kokoro.in"); fi
  if flag_on "$LEGACY_INT8"; then PIP_INSTALL_ARGS+=(-r "$REQ_DIR/legacy-int8.in"); fi
  PIP_INSTALL_ARGS+=(-c "$CONSTRAINTS" --extra-index-url "$PYTORCH_INDEX" --extra-index-url "$NVIDIA_INDEX_URL")
  if [[ -n "$WHEELHOUSE" && -d "$WHEELHOUSE" ]]; then
    PIP_INSTALL_ARGS+=(--find-links "$WHEELHOUSE")
  fi
  SELFTEST_BUCKETS="${HLS_SCHEDULER_FIXED_BATCH_SIZES:-${MUSETALK_TAESD_WARMUP_BATCHES:-8}}"
}

print_plan() {
  printf 'MODE=%s\n' "$MODE"
  printf 'REPO_ROOT=%s\n' "$REPO_ROOT"
  printf 'VENV=%s\n' "$VENV_PATH"
  printf 'VENV_EXISTS=%s\n' "$VENV_EXISTS"
  printf 'VENV_TORCH_TAG=%s\n' "${VENV_TORCH_TAG:-}"
  printf 'MATRIX=%s\n' "$TARGET_MATRIX"
  printf 'MATRIX_REASON=%s\n' "$MATRIX_REASON"
  printf 'GPU=%s\n' "$GPU_DESC"
  printf 'PYTHON_BIN=%s\n' "$PYTHON_BIN"
  printf 'CLEAN=%s\n' "$CLEAN"
  printf 'CONSTRAINTS=%s\n' "$CONSTRAINTS"
  printf 'PYTORCH_INDEX=%s\n' "$PYTORCH_INDEX"
  printf 'GROUP_KOKORO=%s\n' "$KOKORO"
  printf 'GROUP_NATIVE_VP8=%s\n' "$NATIVE_VP8"
  printf 'GROUP_AVATAR_PREP=%s\n' "$AVATAR_PREP"
  printf 'GROUP_LEGACY_INT8=%s\n' "$LEGACY_INT8"
  printf 'GROUP_CHIN_TOOLS=%s\n' "$CHIN_TOOLS"
  printf 'CHIN_VENV=%s\n' "$CHIN_VENV"
  printf 'NATIVE_VP8_DIR=%s\n' "$NATIVE_VP8_DIR"
  printf 'SKIP_WEIGHTS=%s\n' "$SKIP_WEIGHTS"
  printf 'SELFTEST=%s\n' "$(( SELFTEST && GPU_VISIBLE ))"
  printf 'SELFTEST_BUCKETS=%s\n' "$SELFTEST_BUCKETS"
  printf 'STATE_FILE=%s\n' "$STATE_FILE"
  printf 'PIP_COMMAND=%s -m pip' "$VENV_PY"
  printf ' %q' "${PIP_INSTALL_ARGS[@]}"
  printf '\n'
}

if [[ -z "$HOST_PY" ]]; then
  if [[ "$MODE" == "install" && $SKIP_APT -eq 0 && "${EUID:-$(id -u)}" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    step_log_emit INFO "No python >= 3.8 yet: installing system packages first"
    apt_packages_step
    HOST_PY="$(command -v "$PYTHON_BIN" || command -v python3 || true)"
  fi
  [[ -n "$HOST_PY" ]] || die "No python >= 3.8 available and apt will not run; install $PYTHON_BIN first"
fi

if [[ "$MODE" == "plan" ]]; then
  STEP_LOG_SCRIPT_NAME="$SCRIPT_NAME"
  resolve_plan
  print_plan
  exit 0
fi

# ------------------------------------------------------------------------------- install
step_logging_init "$SCRIPT_NAME" "$STEP_LOG_ROOT"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/musetalk-install-XXXXXX")"
LAST_STEP_STATUS=0
ACTIVE_PHASE=""
ACTIVE_PHASE_START=0

on_install_exit() {
  local status="$1"
  if [[ -n "$ACTIVE_PHASE" ]]; then
    step_record_phase_result "$ACTIVE_PHASE" "" "$(( $(date +%s) - ACTIVE_PHASE_START ))" "FAIL"
    step_log_emit ERROR "FAILED: $ACTIVE_PHASE (exit=$status)"
  fi
  rm -rf "$WORK_DIR"
  step_logging_on_exit "$status"
}
trap 'on_install_exit "$?"' EXIT

# Phases are called as plain commands so errexit stays armed. (step_logging.sh's run_phase uses
# `if "$@"`, which silently disables -e for everything nested inside the phase.)
run_install_phase() {
  local label="$1: $2" description="$3"
  shift 3
  ACTIVE_PHASE="$label"
  ACTIVE_PHASE_START="$(date +%s)"
  STEP_LOG_CURRENT_PHASE="$label"
  step_log_emit PHASE "START: $label"
  if [[ -n "$description" ]]; then step_log_emit PHASE "DETAIL: $description"; fi
  "$@"
  local elapsed=$(( $(date +%s) - ACTIVE_PHASE_START ))
  step_record_phase_result "$label" "$description" "$elapsed" "OK"
  step_log_emit PHASE "DONE: $label ($(step_format_duration "$elapsed"))"
  ACTIVE_PHASE=""
  STEP_LOG_CURRENT_PHASE=""
}

# Run a step body in a subshell with errexit really enabled (a bare `if func` would silently
# disable -e inside func). Sets LAST_STEP_STATUS; returns it unless --optional.
run_strict_step() {
  local optional=0
  if [[ "$1" == "--optional" ]]; then optional=1; shift; fi
  local label="$1"; shift
  local qualified="$label" start_ts elapsed status
  if [[ -n "$STEP_LOG_CURRENT_PHASE" ]]; then qualified="$STEP_LOG_CURRENT_PHASE :: $label"; fi
  start_ts="$(date +%s)"
  step_log_emit STEP "START: $qualified"
  trap - ERR
  set +e
  (
    trap 'on_err "$?" "$LINENO" "$BASH_COMMAND"' ERR
    set -Eeuo pipefail
    "$@"
  )
  status=$?
  set -e
  trap 'on_err "$?" "$LINENO" "$BASH_COMMAND"' ERR
  elapsed=$(( $(date +%s) - start_ts ))
  LAST_STEP_STATUS=$status
  if (( status == 0 )); then
    step_record_result "$qualified" "$elapsed" "OK"
    step_log_emit STEP "DONE: $qualified ($(step_format_duration "$elapsed"))"
    return 0
  fi
  if (( optional )); then
    step_record_result "$qualified" "$elapsed" "WARN"
    step_log_emit WARN "OPTIONAL STEP FAILED: $qualified ($(step_format_duration "$elapsed"), exit=$status); continuing"
    return 0
  fi
  step_record_result "$qualified" "$elapsed" "FAIL"
  step_log_emit ERROR "FAILED: $qualified ($(step_format_duration "$elapsed"), exit=$status)"
  return "$status"
}

disk_free_gb() {
  local path="$1"
  while [[ ! -e "$path" && "$path" != "/" ]]; do path="$(dirname "$path")"; done
  df -P -BG "$path" | awk 'NR==2 {gsub("G","",$4); print $4}'
}

write_json() {
  # write_json FILE key=value ... (values: true/false/null/number kept raw, else string)
  local file="$1"; shift
  "$HOST_PY" - "$file" "$@" <<'PY'
import json, re, sys
out = {}
for item in sys.argv[2:]:
    key, _, value = item.partition("=")
    if value in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", value or "x"):
        out[key] = json.loads(value)
    else:
        out[key] = value
json.dump(out, open(sys.argv[1], "w"), indent=2)
PY
}

# ---- phase: preflight
preflight() {
  log "Repo root: $REPO_ROOT"
  log "Venv: $VENV_PATH (exists=$VENV_EXISTS${VENV_TORCH_TAG:+, torch tag $VENV_TORCH_TAG})"
  log "GPU: $GPU_DESC"
  log "Matrix: $TARGET_MATRIX ($MATRIX_REASON)"
  log "Constraints: $CONSTRAINTS"
  log "Groups: kokoro=$KOKORO native_vp8=$NATIVE_VP8 avatar_prep=$AVATAR_PREP legacy_int8=$LEGACY_INT8 chin_tools=$CHIN_TOOLS"
  log "PIP_CACHE_DIR=${PIP_CACHE_DIR:-<pip default>}${WHEELHOUSE:+ wheelhouse=$WHEELHOUSE}"
  if [[ -n "${PYTORCH_INDEX_URL:-}" && "${PYTORCH_INDEX_URL%/}" != "$PYTORCH_INDEX" ]]; then
    warn "Ignoring PYTORCH_INDEX_URL from the environment (it points at another CUDA family); using $PYTORCH_INDEX"
  fi
  [[ -n "${PIP_CONSTRAINT:-}" ]] && warn "PIP_CONSTRAINT is set in the environment; it is applied on top of $CONSTRAINTS"
  [[ -n "${PIP_INDEX_URL:-}" ]] && log "PIP_INDEX_URL is set (mirror); the PyTorch/NVIDIA indexes are added as extra indexes"

  if (( VENV_EXISTS )) && [[ -n "$VENV_TORCH_TAG" && "$VENV_TORCH_TAG" != "$TARGET_MATRIX" ]] && (( CLEAN == 0 )); then
    die "$EXIT_NEEDS_CLEAN" "Venv $VENV_PATH has torch built for $VENV_TORCH_TAG but $TARGET_MATRIX is required ($MATRIX_REASON). Re-run with --clean."
  fi
  local fresh=0 need free
  if (( CLEAN )) || [[ ! -x "$VENV_PY" ]]; then fresh=1; fi
  need=$MIN_DISK_GB_REPAIR
  (( fresh )) && need=$MIN_DISK_GB_FRESH
  free="$(disk_free_gb "$VENV_PATH")"
  local reclaim=0
  if (( CLEAN )) && [[ -f "$VENV_PATH/pyvenv.cfg" ]]; then
    # --clean deletes the old venv before installing, so its size counts as free space here
    reclaim="$(du -sk "$VENV_PATH" 2>/dev/null | awk '{printf "%d", $1 / 1048576}')"
    reclaim="${reclaim:-0}"
  fi
  log "Disk free at the venv location: ${free} GB$( (( reclaim > 0 )) && echo " + ${reclaim} GB from the venv --clean replaces") (need >= ${need} GB for a $( (( fresh )) && echo fresh || echo repair ) install)"
  if [[ -n "$free" ]] && (( free + reclaim < need )); then
    die "Not enough disk: ${free} GB free, ${need} GB required (override with MUSETALK_INSTALL_MIN_DISK_GB)"
  fi
}

# ---- phase: apt

phase_apt() {
  if (( SKIP_APT )); then
    log "Skipping apt packages (--skip-apt)"
    return 0
  fi
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    warn "Not root: skipping apt packages (python3.10-venv, ffmpeg, coturn, libgl1 must already be present)"
    return 0
  fi
  command -v apt-get >/dev/null 2>&1 || { warn "apt-get not found; skipping system packages"; return 0; }
  run_strict_step "Install system packages (apt)" apt_packages_step
}

# ---- phase: venv
remove_venv_step() {
  [[ "$VENV_PATH" == /* && "$VENV_PATH" != "/" && "$VENV_PATH" != "$WORKSPACE_ROOT" && "$VENV_PATH" != "$HOME" ]] || {
    echo "refusing to delete suspicious venv path: $VENV_PATH" >&2; return 1; }
  [[ -f "$VENV_PATH/pyvenv.cfg" ]] || { echo "refusing to delete $VENV_PATH: no pyvenv.cfg (not a venv)" >&2; return 1; }
  rm -rf "$VENV_PATH"
}

create_venv_step() {
  mkdir -p "$(dirname "$VENV_PATH")"
  "$PYTHON_BIN" -m venv "$VENV_PATH"
  [[ -x "$VENV_PY" ]]
  "$VENV_PY" -m pip install --disable-pip-version-check "pip==$PIP_VERSION" wheel setuptools -c "$CONSTRAINTS"
}

upgrade_old_pip_step() {
  "$VENV_PY" -m pip install --disable-pip-version-check "pip==$PIP_VERSION"
}

phase_venv() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Base interpreter not found: $PYTHON_BIN"
  "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)' \
    || die "$PYTHON_BIN is not CPython 3.10 (the pinned stack needs 3.10)"
  "$PYTHON_BIN" -c 'import ensurepip, venv' 2>/dev/null || die "$PYTHON_BIN lacks venv/ensurepip (apt install python3.10-venv)"
  if (( CLEAN )) && [[ -e "$VENV_PATH" ]]; then
    run_strict_step "Remove existing venv (--clean)" remove_venv_step
  fi
  if [[ -e "$VENV_PATH" && ! -x "$VENV_PY" ]]; then
    die "$EXIT_NEEDS_CLEAN" "$VENV_PATH exists but has no bin/python; re-run with --clean"
  fi
  if [[ ! -x "$VENV_PY" ]]; then
    run_strict_step "Create venv ($PYTHON_BIN)" create_venv_step
  else
    log "Reusing existing venv (repair in place): $VENV_PATH"
    local pip_major
    pip_major="$("$VENV_PY" -c 'import pip; print(pip.__version__.split(".")[0])' 2>/dev/null || echo 0)"
    if (( pip_major < 23 )); then
      run_strict_step "Upgrade pip $pip_major.x -> $PIP_VERSION" upgrade_old_pip_step
    fi
  fi
}

# ---- phase: python packages
pip_install_groups_step() {
  "$VENV_PY" -m pip "${PIP_INSTALL_ARGS[@]}"
}

avatar_prep_step() {
  local -a common=(-c "$CONSTRAINTS" --extra-index-url "$PYTORCH_INDEX" --extra-index-url "$NVIDIA_INDEX_URL")
  # openmim is NOT installed up front: its dependency chain (opendatalab -> openxlab) requires rich~=13.4.2, while
  # the constraints pin rich 15 (Kokoro's typer needs rich>=13.8), so pip stopped with ResolutionImpossible and every
  # fresh --with-avatar-prep install failed. Nothing imports mim/opendatalab/openxlab; mmcv comes from the repo wheel.
  "$VENV_PY" -m pip install --disable-pip-version-check "setuptools<81" ninja psutil "${common[@]}"
  local have_mmcv wheel=""
  have_mmcv="$("$VENV_PY" -c 'import importlib.metadata as m; print(m.version("mmcv"))' 2>/dev/null || true)"
  if [[ "$have_mmcv" == "$MMCV_VERSION" ]]; then
    log "mmcv $MMCV_VERSION already installed"
  else
    wheel="$(find "$REPO_ROOT/third_party_wheels/mmcv" -maxdepth 1 -type f -name "mmcv-${MMCV_VERSION}-cp310-*.whl" 2>/dev/null | sort -r | head -n 1 || true)"
    if [[ -n "$wheel" ]]; then
      log "Installing mmcv from the repo-local wheel: $wheel"
      "$VENV_PY" -m pip install --disable-pip-version-check --no-deps "$wheel"
    elif "$VENV_PY" -m pip install --disable-pip-version-check --no-deps openmim "${common[@]}" \
        && "$VENV_PY" -m pip install --disable-pip-version-check click colorama model-index tabulate "${common[@]}" \
        && "$VENV_PY" -m mim install "mmcv==$MMCV_VERSION"; then
      # (mim only for this fallback, without the opendatalab/openxlab chain that cannot resolve; see above)
      log "Installed mmcv $MMCV_VERSION via mim (OpenMMLab prebuilt wheel)"
    else
      local torch_cuda nvcc_cuda
      torch_cuda="$(CUDA_VISIBLE_DEVICES="" "$VENV_PY" -c 'import torch; print(torch.version.cuda or "")')"
      nvcc_cuda="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1 || true)"
      [[ -n "$nvcc_cuda" && "$nvcc_cuda" == "$torch_cuda" ]] || {
        echo "mmcv source build needs nvcc $torch_cuda (found: ${nvcc_cuda:-none})" >&2; return 1; }
      MAX_JOBS="$MMCV_BUILD_MAX_JOBS" "$VENV_PY" -m pip install --disable-pip-version-check --no-build-isolation "mmcv==$MMCV_VERSION"
    fi
  fi
  "$VENV_PY" -m pip install --disable-pip-version-check --no-build-isolation chumpy -c "$CONSTRAINTS"
  "$VENV_PY" -m pip install --disable-pip-version-check mmengine mmdet mmpose "${common[@]}"
  "$VENV_PY" -B "$REPO_ROOT/scripts/patch_mmengine_compat.py"
  CUDA_VISIBLE_DEVICES="" "$VENV_PY" -B -c 'import mmengine, mmcv, mmcv._ext, mmcv.ops, mmdet, mmpose; print("avatar-prep imports OK", mmcv.__version__, mmdet.__version__, mmpose.__version__)'
}

phase_packages() {
  log "pip:$(printf ' %q' "${PIP_INSTALL_ARGS[@]}")"
  run_strict_step "Install server packages (server.in$(flag_on "$KOKORO" && echo ' + kokoro.in')$(flag_on "$LEGACY_INT8" && echo ' + legacy-int8.in'))" pip_install_groups_step
  if flag_on "$AVATAR_PREP"; then
    run_strict_step "Install avatar-prep stack (mmcv/mmdet/mmpose)" avatar_prep_step
  fi
}

# ---- phase: weights
download_weights_step() {
  cd "$REPO_ROOT"
  PATH="$VENV_PATH/bin:$PATH" \
  DOWNLOAD_MUSETALK_V1_WEIGHTS=0 \
  DOWNLOAD_AVATAR_PREP_WEIGHTS="$AVATAR_PREP" \
  DOWNLOAD_SYNCNET_WEIGHTS="${DOWNLOAD_SYNCNET_WEIGHTS:-0}" \
  DOWNLOAD_TAESD_WEIGHTS=1 \
  DOWNLOAD_KOKORO_WEIGHTS="$KOKORO" \
  PIP_CONSTRAINT="$CONSTRAINTS" \
    bash "$REPO_ROOT/download_weights.sh"
}

phase_weights() {
  if (( SKIP_WEIGHTS )); then
    log "Skipping weight downloads (--skip-weights)"
    return 0
  fi
  run_strict_step "Download weights (MuseTalk V1.5, SD-VAE, Whisper, face-parse, TAESD$(flag_on "$KOKORO" && echo ', Kokoro cache'))" download_weights_step
}

# ---- phase: native VP8
native_vp8_step() {
  local -a install_args=(--directory "$NATIVE_VP8_DIR")
  [[ -n "$NATIVE_VP8_WHEEL" ]] && install_args+=(--wheel "$NATIVE_VP8_WHEEL")
  write_json "$WORK_DIR/native_vp8.json" enabled=true "directory=$NATIVE_VP8_DIR" installed=false verified=false preflight_ok=false
  "$VENV_PY" -B "$REPO_ROOT/scripts/install_native_vp8.py" "${install_args[@]}"
  "$VENV_PY" -B "$REPO_ROOT/scripts/install_native_vp8.py" --directory "$NATIVE_VP8_DIR" --verify
  write_json "$WORK_DIR/native_vp8.json" enabled=true "directory=$NATIVE_VP8_DIR" installed=true verified=true preflight_ok=false
  (
    cd "$REPO_ROOT"
    WEBRTC_VP8_ENCODER=native WEBRTC_NATIVE_VP8_DIR="$NATIVE_VP8_DIR" CUDA_VISIBLE_DEVICES="" \
      "$VENV_PY" -B -c "from scripts import webrtc_native_vp8 as n; n.configure_vp8_encoder('preflight')"
  )
  write_json "$WORK_DIR/native_vp8.json" enabled=true "directory=$NATIVE_VP8_DIR" installed=true verified=true preflight_ok=true
}

NATIVE_VP8_FINAL=0
phase_native_vp8() {
  if ! flag_on "$NATIVE_VP8"; then
    log "Native VP8 not requested (x86_64 + CPython 3.10 only); WEBRTC_VP8_ENCODER stays pyav"
    write_json "$WORK_DIR/native_vp8.json" enabled=false
    return 0
  fi
  if (( NATIVE_VP8_EXPLICIT )); then
    run_strict_step "Install + verify + preflight native VP8" native_vp8_step
  else
    run_strict_step --optional "Install + verify + preflight native VP8 (auto)" native_vp8_step
  fi
  if (( LAST_STEP_STATUS != 0 )); then
    # auto mode only (explicit failures already stopped the install): do not claim the group in the
    # stamp, so --check does not demand a repair for an optional encoder.
    warn "Native VP8 unavailable; the fast recipe keeps WEBRTC_VP8_ENCODER=pyav (default anyway)"
    write_json "$WORK_DIR/native_vp8.json" enabled=false attempted=true verified=false preflight_ok=false \
      "directory=$NATIVE_VP8_DIR" "error=optional native VP8 install failed (exit $LAST_STEP_STATUS); see the install log"
    return 0
  fi
  NATIVE_VP8_FINAL=1
}

# ---- phase: chin tools (dedicated venv; never the server venv)
chin_tools_step() {
  if [[ ! -x "$CHIN_VENV/bin/python" ]]; then
    mkdir -p "$(dirname "$CHIN_VENV")"
    "$PYTHON_BIN" -m venv "$CHIN_VENV"
    "$CHIN_VENV/bin/python" -m pip install --disable-pip-version-check "pip==$PIP_VERSION"
  fi
  "$CHIN_VENV/bin/python" -m pip install --disable-pip-version-check \
    -r "$REQ_DIR/chin-tools.in" -c "$REQ_DIR/constraints-chin-tools.txt"
  CUDA_VISIBLE_DEVICES="" "$CHIN_VENV/bin/python" -B - <<'PY'
import cv2, numpy, mediapipe as mp
mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
mesh.close()
print("chin tools OK: mediapipe", mp.__version__, "numpy", numpy.__version__, "opencv", cv2.__version__)
PY
}

phase_chin_tools() {
  if ! flag_on "$CHIN_TOOLS"; then
    return 0
  fi
  log "Chin tools venv: $CHIN_VENV (exported as MUSETALK_CHIN_TRACKER_PYTHON via the install stamp)"
  run_strict_step "Install dedicated chin-tracker venv (mediapipe 0.10.9)" chin_tools_step
}

# ---- phase: smoke + self-test
smoke_step() {
  run_import_smoke "$WORK_DIR/smoke.json" "$KOKORO" "$LEGACY_INT8" "$AVATAR_PREP"
}

selftest_step() {
  local -a args=(--repo-root "$REPO_ROOT" --out "$SELFTEST_OUT" --buckets "$SELFTEST_BUCKETS")
  (( SELFTEST_UNET )) && args+=(--unet)
  mkdir -p "$(dirname "$SELFTEST_OUT")"
  cd "$REPO_ROOT"
  timeout "$SELFTEST_TIMEOUT_S" "$VENV_PY" -B "$REPO_ROOT/scripts/musetalk_selftest.py" "${args[@]}"
}

phase_verify() {
  run_strict_step "CPU import smoke (CUDA hidden)" smoke_step
  local deferred
  deferred="$("$HOST_PY" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1])).get("deferred") or []))' "$WORK_DIR/smoke.json" 2>/dev/null || true)"
  if [[ -n "$deferred" ]]; then
    warn "Imports deferred (they need the NVIDIA driver or a visible GPU; the GPU self-test imports them): $deferred"
  fi
  if (( SELFTEST == 0 )); then
    log "GPU self-test skipped (--no-selftest)"
  elif [[ "$GPU_VISIBLE" != "1" ]]; then
    log "GPU self-test skipped: no GPU visible (image build?). Run later: $VENV_PY scripts/musetalk_selftest.py"
  else
    run_strict_step --optional "GPU self-test (TAESD compile/timing, TRT import, engine keys)" selftest_step
    (( LAST_STEP_STATUS == 0 )) || warn "GPU self-test failed; see $SELFTEST_OUT. The resolver treats a missing/failed self-test as 'unknown'."
  fi
}

# ---- phase: stamp + final check
stamp_step() {
  local groups
  groups="$("$HOST_PY" -c 'import json,sys; print(json.dumps({"server": True, "kokoro": sys.argv[1]=="1", "avatar_prep": sys.argv[2]=="1", "legacy_int8": sys.argv[3]=="1", "chin_tools": sys.argv[4]=="1"}))' \
    "$KOKORO" "$AVATAR_PREP" "$LEGACY_INT8" "$CHIN_TOOLS")"
  local -a args=(stamp --repo-root "$REPO_ROOT" --venv "$VENV_PATH" --matrix "$TARGET_MATRIX"
                 --python-bin "$PYTHON_BIN" --groups-json "$groups" --chin-venv "$CHIN_VENV"
                 --native-vp8-json "$WORK_DIR/native_vp8.json" --smoke-json "$WORK_DIR/smoke.json"
                 --out "$STATE_FILE")
  if [[ -f "$SELFTEST_OUT" ]]; then
    args+=(--selftest-json "$SELFTEST_OUT" --selftest-path "$SELFTEST_OUT")
  fi
  state_helper "${args[@]}"
}

final_check_step() {
  local -a args=(check --repo-root "$REPO_ROOT" --venv "$VENV_PATH" --matrix "$TARGET_MATRIX"
                 --state-file "$STATE_FILE" --native-vp8-dir "$NATIVE_VP8_DIR" --chin-venv "$CHIN_VENV"
                 --kokoro "$KOKORO" --avatar-prep "$AVATAR_PREP" --legacy-int8 "$LEGACY_INT8"
                 --chin-tools "$CHIN_TOOLS")
  args+=(--native-vp8 "$NATIVE_VP8_FINAL")
  (( SKIP_WEIGHTS )) && args+=(--skip-weights)
  state_helper "${args[@]}"
}

resolve_plan
run_install_phase "Phase 0" "Preflight" "Resolve matrix, groups and disk budget; refuse a matrix change without --clean." preflight
run_install_phase "Phase 1" "System packages" "apt: python3.10(+venv,dev), ffmpeg, coturn, build tools, GL libs (root only)." phase_apt
run_install_phase "Phase 2" "Venv" "Create the CPython 3.10 venv, or reuse it and repair in place." phase_venv
run_install_phase "Phase 3" "Python packages" "One pip resolve of the selected requirements/*.in against constraints-$TARGET_MATRIX.txt." phase_packages
run_install_phase "Phase 4" "Weights" "download_weights.sh (+ pinned TAESD, optional Kokoro cache)." phase_weights
run_install_phase "Phase 5" "Native VP8" "Pinned native VP8 encoder files + CPU preflight (default encoder stays pyav)." phase_native_vp8
run_install_phase "Phase 6" "Chin tools" "Optional dedicated MediaPipe venv for the offline chin tracker." phase_chin_tools
run_install_phase "Phase 7" "Verify" "CPU import smoke with CUDA hidden; GPU self-test when a GPU is visible." phase_verify
run_install_phase "Phase 8" "Stamp" "Write the install stamp, then re-run the read-only check." run_strict_step "Write $STATE_FILE" stamp_step
run_install_phase "Phase 9" "Post-install check" "musetalk_install_state.py check with the resolved groups (must exit 0)." run_strict_step "Post-install --check" final_check_step
log "MuseTalk install complete: venv=$VENV_PATH matrix=$TARGET_MATRIX stamp=$STATE_FILE"
log "Start the server with: bash $REPO_ROOT/scripts/run_musetalk_server.sh (recipe fast by default)"
