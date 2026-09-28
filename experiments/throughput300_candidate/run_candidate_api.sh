#!/usr/bin/env bash
# Candidate API launcher for the 300 fps plan (docs/musetalk_4070s_300fps_plan_2026-09-27.md, section 4).
#
# * Runs THIS worktree's api_server.py (/workspace/MuseTalk-perf300), port 8300 by default.
# * Sources the generated .runtime/musetalk_trt_local_sm89.env ("Do not edit by hand"), then the
#   live launcher's exports (copied below from
#   /workspace/experiments/chinese_bob_webrtc_20260927/run_local_api.sh, which is neither read
#   nor modified here), then the lever overlay musetalk_300fps.env (one lever per line).
# * oom_score_adj 1000, its own log, server CPUs pinned away from the load-test client cores
#   (client = physical cores 12-15 = CPUs 12-15,28-31; server = the rest).
# * Local Kokoro TTS is disabled (MUSETALK_DISABLE_LOCAL_TTS=1) in both modes.
# * It never edits or execs any existing start script.
#
# Usage:
#   run_candidate_api.sh                 # candidate: generated env + live exports + overlay
#   MUSETALK_300FPS_OVERLAY=none run_candidate_api.sh
#                                        # baseline: today's defaults (only the measurement-only
#                                        # WEBRTC_LIFETIME_COUNTERS=1 and the TTS disable are added)
#   MUSETALK_API_PORT=8301 MUSETALK_300FPS_OVERLAY=/path/other.env run_candidate_api.sh
# Env knobs: MUSETALK_API_HOST (127.0.0.1), MUSETALK_API_PORT (8300), MUSETALK_300FPS_OVERLAY
#   (default: musetalk_300fps.env next to this script; "none" = baseline), MUSETALK_CANDIDATE_LOG,
#   MUSETALK_SERVER_CPUS (0-11,16-27; "all" = no pinning), MUSETALK_CANDIDATE_KEEP_STUN (0).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=${MUSETALK_PYTHON:-/workspace/.venvs/musetalk_trt_stagewise/bin/python}
HOST=${MUSETALK_API_HOST:-127.0.0.1}
PORT=${MUSETALK_API_PORT:-8300}
OVERLAY=${MUSETALK_300FPS_OVERLAY:-$SCRIPT_DIR/musetalk_300fps.env}
SERVER_CPUS=${MUSETALK_SERVER_CPUS:-0-11,16-27}
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
MODE=candidate
[[ "$OVERLAY" == "none" ]] && MODE=baseline
LOG=${MUSETALK_CANDIDATE_LOG:-$LOG_DIR/api_${MODE}_${PORT}_$(date -u +%Y%m%dT%H%M%SZ).log}

die() { echo "[run_candidate_api] ERROR: $*" >&2; exit 1; }

[[ "$REPO_ROOT" == /workspace/MuseTalk-perf300 ]] || die "expected the perf300 worktree, got $REPO_ROOT"
[[ -f "$REPO_ROOT/api_server.py" ]] || die "api_server.py missing in $REPO_ROOT"
if [[ "$PORT" == 8000 || "$PORT" == 8200 ]]; then
  die "port $PORT belongs to the user's live server; use 8300 (default)"
fi
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  die "port $PORT is already in use"
fi

cd "$REPO_ROOT"
set -a
# shellcheck disable=SC1091
source "$REPO_ROOT/.runtime/musetalk_trt_local_sm89.env"
set +a

# Live launcher exports (copied from run_local_api.sh on 2026-09-28).
export MUSETALK_VAE_BACKEND=taesd
export MUSETALK_UNET_BACKEND=trt
export MUSETALK_TRT_FALLBACK=0
export MUSETALK_BLEND_FIXED_POINT=1
export MUSETALK_BLEND_SHRINK_MASK_BBOX=1
export MUSETALK_TAESD_WARMUP_BATCHES=8
export WEBRTC_MOTION_ATLAS=/workspace/experiments/chinese_bob_webrtc_20260927/motion-atlas.json
export WEBRTC_MOTION_ALLOW_UNREVIEWED=1
export AVATAR_S3_ENABLED=0
export KOKORO_TTS_DEVICE=cpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Rig conditions shared by baseline and candidate (not levers).
export MUSETALK_DISABLE_LOCAL_TTS=1        # plan 0.5: no local Kokoro under load
export WEBRTC_LIFETIME_COUNTERS=1          # measurement only: counters + send ring for the harness
if [[ "${MUSETALK_CANDIDATE_KEEP_STUN:-0}" != 1 ]]; then
  export WEBRTC_STUN_URLS=""               # loopback rig: host ICE candidates only
fi

if [[ "$MODE" == candidate ]]; then
  [[ -f "$OVERLAY" ]] || die "overlay $OVERLAY not found"
  set -a
  # shellcheck disable=SC1090
  source "$OVERLAY"
  set +a
fi

{
  echo "[run_candidate_api] $(date -u +%FT%TZ) mode=$MODE host=$HOST port=$PORT pid=$$"
  echo "[run_candidate_api] repo=$REPO_ROOT python=$PY server_cpus=$SERVER_CPUS"
  echo "[run_candidate_api] overlay=$OVERLAY"
  if [[ "$MODE" == candidate ]]; then
    grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$OVERLAY" | sed 's/^/[run_candidate_api]   lever /'
  fi
  echo "[run_candidate_api] log=$LOG"
} | tee -a "$LOG" >&2

echo 1000 > /proc/self/oom_score_adj 2>/dev/null || echo "[run_candidate_api] warning: could not set oom_score_adj" >&2

CMD=("$PY" api_server.py --host "$HOST" --port "$PORT")
if [[ "$SERVER_CPUS" != all ]] && command -v taskset >/dev/null 2>&1; then
  CMD=(taskset -c "$SERVER_CPUS" "${CMD[@]}")
fi
exec "${CMD[@]}" >>"$LOG" 2>&1
