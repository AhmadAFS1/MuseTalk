#!/usr/bin/env bash
# Vast.ai on-start for the character-creation worker (MiniMax H3 three-pose + MuseTalk caches).
#
# Runs from a MuseTalk checkout (the template's on-start clones it, then execs this file).
# Order: preflight -> start the worker API (reports the install phase on /health at once)
#        -> install (MuseTalk + avatar prep, runtime secret, pinned H3 runtime) -> verify -> ready.
# Every exit path prints exactly one CHARACTER_WORKER_ONSTART COMPLETE or FAILED marker and
# leaves $ROOT/status.json describing the phase, so the orchestrator can tell "booting" from
# "broken" without SSH. Re-running is safe: every install step is idempotent.
#
# Environment (template env or the AWS runtime secret):
#   CHARACTER_WORKER_PORT=8010         CHARACTER_WORKER_TOKEN=<bearer token for the API>
#   MUSETALK_AWS_SECRET_ID + AWS_* secret-reader key (docs/musetalk_worker_secrets.md)
#   H3_WORKFLOWS_GIT_TOKEN|GITHUB_TOKEN  read access to the private minimax-h3-workflows repo
#   CHARACTER_WORKER_CHIN_TOOLS=1      also build the MediaPipe venv (landmark QC); default 1
#   CHARACTER_WORKER_ALLOW_UNTESTED_GPU=1  proceed on a GPU outside the lock's tested range
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE="${WORKSPACE:-/workspace}"
ROOT="${CHARACTER_WORKER_ROOT:-$WORKSPACE/character_worker}"
PORT="${CHARACTER_WORKER_PORT:-8010}"
LOCK_JSON="$SCRIPT_DIR/locks/character_worker.lock.json"
STATUS_FILE="$ROOT/status.json"
ONSTART_LOG="${CHARACTER_WORKER_ONSTART_LOG:-$ROOT/onstart.log}"
START_TS="$(date +%s)"
MARKER_PRINTED=0
MAIN_PID="$$"
export WORKSPACE REPO_ROOT CHARACTER_WORKER_ROOT="$ROOT" CHARACTER_WORKER_PORT="$PORT"

mkdir -p "$ROOT/logs"

plan() {
  cat <<PLAN
character worker on-start plan
  repo:        $REPO_ROOT
  workspace:   $WORKSPACE
  state root:  $ROOT (status.json, runtime.env, versions.json, jobs/)
  api:         python3 $SCRIPT_DIR/character_api.py --host 0.0.0.0 --port $PORT  (started first, serves /health while installing)
  install:     bash $SCRIPT_DIR/install_character_worker.sh  (MuseTalk --with-avatar-prep [--with-chin-tools], runtime secret, pinned H3)
  lock:        $LOCK_JSON
  markers:     CHARACTER_WORKER_ONSTART BEGIN / COMPLETE / FAILED in $ONSTART_LOG
PLAN
}
if [[ "${1:-}" == "--plan" ]]; then plan; exit 0; fi
[[ $# -eq 0 ]] || { echo "usage: $0 [--plan]" >&2; exit 2; }

exec > >(tee -a "$ONSTART_LOG") 2>&1
echo "========================================"
echo "CHARACTER_WORKER_ONSTART BEGIN: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"

log() { printf '[onstart] [%s] %s\n' "$(date -u '+%H:%M:%S')" "$*"; }
elapsed() { local s=$(( $(date +%s) - START_TS )); printf '%dm%02ds' $((s / 60)) $((s % 60)); }

write_status() {
  # write_status <phase> <ready 0|1> <failed 0|1> <message>
  python3 - "$STATUS_FILE" "$1" "$2" "$3" "$4" "$START_TS" <<'PY'
import json, os, sys, time
path, phase, ready, failed, message, started = sys.argv[1:]
data = {"service": "character-worker", "phase": phase, "ready": ready == "1", "failed": failed == "1", "message": message,
        "started_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(int(started))),
        "updated_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), "elapsed_seconds": int(time.time()) - int(started),
        "instance_id": os.environ.get("VAST_INSTANCE_ID"), "log": os.environ.get("CHARACTER_WORKER_ONSTART_LOG")}
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(data, f, indent=2)
os.replace(tmp, path)
PY
}

print_failed() {
  [[ "${BASHPID:-$$}" == "$MAIN_PID" ]] || return 0
  (( MARKER_PRINTED )) && return 0
  MARKER_PRINTED=1
  echo "========================================"
  echo "CHARACTER_WORKER_ONSTART FAILED: $(date -u '+%Y-%m-%d %H:%M:%S UTC')  (elapsed $(elapsed))"
  echo "LOG FILE: $ONSTART_LOG"
  echo "========================================"
}
die() {
  log "ERROR: $*"
  write_status "${CURRENT_PHASE:-unknown}" 0 1 "$*" || true
  trap - ERR
  print_failed
  exit 1
}
trap 'status=$?; [[ "${BASHPID:-$$}" == "$MAIN_PID" ]] || exit $status; trap - ERR; log "unhandled failure near line ${BASH_LINENO[0]:-?} (exit $status)"; write_status "${CURRENT_PHASE:-unknown}" 0 1 "unhandled failure near line ${BASH_LINENO[0]:-?} (exit $status)" || true; print_failed; exit $status' ERR
trap 'status=$?; if (( status != 0 )) && [[ "${BASHPID:-$$}" == "$MAIN_PID" ]] && (( ! MARKER_PRINTED )); then print_failed; fi' EXIT

exec 9>"$ROOT/onstart.lock"
if ! flock -n 9; then
  log "another on-start is already running; exiting"
  trap - EXIT
  exit 0
fi

# ---------------------------------------------------------------- preflight
CURRENT_PHASE=preflight
write_status preflight 0 0 "checking host requirements"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found; this worker needs an NVIDIA GPU host"
command -v python3 >/dev/null || die "python3 not found"
[[ -f "$LOCK_JSON" ]] || die "lock file missing: $LOCK_JSON"
mapfile -t HOST_REQ < <(python3 - "$LOCK_JSON" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))["host_requirements"]
print(h["min_vram_mib"]); print(h["min_ram_mib"]); print(h["min_free_disk_gib_fresh"]); print(h["min_free_disk_gib_resume"])
print(h["min_driver_version"]); print(h["compute_capability"]["min"]); print(h["compute_capability"]["max_exclusive"])
PY
)
MIN_VRAM="${HOST_REQ[0]}"; MIN_RAM="${HOST_REQ[1]}"; MIN_DISK_FRESH="${HOST_REQ[2]}"; MIN_DISK_RESUME="${HOST_REQ[3]}"
MIN_DRIVER="${HOST_REQ[4]}"; CAP_MIN="${HOST_REQ[5]}"; CAP_MAX="${HOST_REQ[6]}"
GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader,nounits | head -n1)"
GPU_NAME="$(echo "$GPU_LINE" | cut -d, -f1 | xargs)"; GPU_VRAM="$(echo "$GPU_LINE" | cut -d, -f2 | xargs)"
GPU_DRIVER="$(echo "$GPU_LINE" | cut -d, -f3 | xargs)"; GPU_CAP="$(echo "$GPU_LINE" | cut -d, -f4 | xargs)"
RAM_MIB="$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo)"
FREE_GIB="$(df -Pk "$WORKSPACE" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
log "GPU: $GPU_NAME ${GPU_VRAM} MiB, driver $GPU_DRIVER, compute $GPU_CAP; RAM ${RAM_MIB} MiB; free disk ${FREE_GIB} GiB"
(( ${GPU_VRAM%.*} >= MIN_VRAM )) || die "GPU has ${GPU_VRAM} MiB; the H3 recipe needs >= ${MIN_VRAM} MiB"
(( RAM_MIB >= MIN_RAM )) || die "host RAM ${RAM_MIB} MiB < ${MIN_RAM} MiB (H3 peaked at 27.6 GiB on the reference run)"
dpkg --compare-versions "$GPU_DRIVER" ge "$MIN_DRIVER" || die "driver $GPU_DRIVER < $MIN_DRIVER (the pinned H3 stack is torch cu130)"
if ! python3 -c "import sys; cap=float('$GPU_CAP'); sys.exit(0 if $CAP_MIN <= cap < $CAP_MAX else 1)"; then
  if [[ "${CHARACTER_WORKER_ALLOW_UNTESTED_GPU:-0}" == 1 ]]; then
    log "WARNING: compute capability $GPU_CAP is outside the tested range [$CAP_MIN, $CAP_MAX); continuing because CHARACTER_WORKER_ALLOW_UNTESTED_GPU=1"
  else
    die "compute capability $GPU_CAP is outside the tested range [$CAP_MIN, $CAP_MAX); set CHARACTER_WORKER_ALLOW_UNTESTED_GPU=1 to try anyway"
  fi
fi
NEED_DISK="$MIN_DISK_FRESH"
[[ -f "$ROOT/h3_install.stamp" && -x "$WORKSPACE/.venvs/musetalk_trt_stagewise/bin/python" ]] && NEED_DISK="$MIN_DISK_RESUME"
(( FREE_GIB >= NEED_DISK )) || die "only ${FREE_GIB} GiB free on $WORKSPACE; need ${NEED_DISK} GiB (61 GB H3 weights + two venvs + working space)"

# ---------------------------------------------------------------- worker API first
CURRENT_PHASE=api
write_status api 0 0 "starting the worker API on port $PORT"
api_listening() { python3 - "$PORT" <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(1)
try: s.connect(("127.0.0.1", int(sys.argv[1]))); print("yes")
except OSError: print("no")
finally: s.close()
PY
}
if [[ "$(api_listening)" == yes ]]; then
  log "something already listens on :$PORT; assuming an earlier worker API (restart it by hand if it is stale)"
else
  nohup python3 "$SCRIPT_DIR/character_api.py" --host 0.0.0.0 --port "$PORT" --root "$ROOT" --workspace "$WORKSPACE" \
    >> "$ROOT/logs/character_api.log" 2>&1 &
  echo $! > "$ROOT/character_api.pid"
  for _ in $(seq 1 20); do [[ "$(api_listening)" == yes ]] && break; sleep 0.5; done
  [[ "$(api_listening)" == yes ]] || die "worker API did not start; see $ROOT/logs/character_api.log"
  log "worker API up (pid $(cat "$ROOT/character_api.pid")); /health reports the install phase from now on"
fi

# ---------------------------------------------------------------- install
CURRENT_PHASE=install
write_status install 0 0 "installing MuseTalk (avatar prep), runtime secret and the pinned H3 runtime"
phase_start=$(date +%s)
CHARACTER_WORKER_STATUS_FILE="$STATUS_FILE" bash "$SCRIPT_DIR/install_character_worker.sh" || die "installer failed; see $ONSTART_LOG and $ROOT/logs/"
log "install phase finished in $(( $(date +%s) - phase_start ))s"

# ---------------------------------------------------------------- verify
CURRENT_PHASE=verify
write_status verify 0 0 "verifying the pipeline preflight"
problems="$(python3 "$SCRIPT_DIR/pipeline.py" --preflight 2>&1 || true)"
if [[ -n "$problems" ]]; then die "pipeline preflight failed: $problems"; fi
[[ "$(api_listening)" == yes ]] || die "worker API is no longer listening on :$PORT"
curl -fsS --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || log "WARNING: /health did not answer within 10 s (curl missing?)"

CURRENT_PHASE=ready
write_status ready 1 0 "ready: POST /createCharacter on port $PORT"
log "versions: $(cat "$ROOT/versions.json" 2>/dev/null | tr -d '\n' | cut -c1-400)"
MARKER_PRINTED=1
trap - ERR EXIT
echo "========================================"
echo "CHARACTER_WORKER_ONSTART COMPLETE: $(date -u '+%Y-%m-%d %H:%M:%S UTC')  (elapsed $(elapsed))"
echo "WORKER API: http://0.0.0.0:$PORT  (mapped port: ${VAST_TCP_PORT_8010:-unknown}, public ip: ${PUBLIC_IPADDR:-unknown})"
echo "LOG FILE: $ONSTART_LOG"
echo "========================================"
