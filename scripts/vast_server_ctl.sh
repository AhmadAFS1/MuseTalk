#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE:-/workspace}"
VENV_PATH="${VENV_PATH:-$WORKSPACE_ROOT/.venvs/musetalk_trt_stagewise}"
PROFILE="${PROFILE:-throughput_record}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"
HEALTH_URL="${HEALTH_URL:-http://${HEALTH_HOST}:${PORT}/health}"
DRAIN_URL="${DRAIN_URL:-http://${HEALTH_HOST}:${PORT}/worker/drain}"
WORKER_STATE_URL="${WORKER_STATE_URL:-http://${HEALTH_HOST}:${PORT}/worker/state}"
LOG_DIR="${LOG_DIR:-$WORKSPACE_ROOT/logs/musetalk}"
PID_FILE="${PID_FILE:-$LOG_DIR/api_server_${PORT}.pid}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/api_server_${PORT}.log}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-600}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-2}"
DRAIN_TIMEOUT_SECONDS="${DRAIN_TIMEOUT_SECONDS:-300}"
TURN_ENV_FILE="${TURN_ENV_FILE:-$REPO_ROOT/.env.webrtc-turn.local}"
TURN_LOG_FILE="${TURN_LOG_FILE:-$LOG_DIR/turnserver.log}"
TURN_PID_FILE="${TURN_PID_FILE:-$LOG_DIR/turnserver.pid}"
VAST_SERVER_CTL_LOAD_TURN_ENV="${VAST_SERVER_CTL_LOAD_TURN_ENV:-1}"
WEBRTC_RELAY_ENABLED="${WEBRTC_RELAY_ENABLED:-0}"
WEBRTC_TURN_AUTOSTART="${WEBRTC_TURN_AUTOSTART:-0}"
TURN_ENV_LOADED=0

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
}

format_duration() {
  local total_seconds="${1:-0}"
  local minutes=$((total_seconds / 60))
  local seconds=$((total_seconds % 60))

  if (( minutes > 0 )); then
    printf '%dm%02ds' "$minutes" "$seconds"
  else
    printf '%ss' "$seconds"
  fi
}

die() {
  printf '[%s] ERROR: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME <start|stop|restart|status|logs>

Control the current MuseTalk TRT-stagewise server for Vast.ai/Jupyter-style
instances without relying on interactive shells.

Environment:
  REPO_ROOT                  MuseTalk repo root (default: $REPO_ROOT)
  VENV_PATH                  TRT-stagewise venv (default: $VENV_PATH)
  PROFILE                    baseline, throughput_record, or vram_max (default: $PROFILE)
  HOST                       Bind host (default: $HOST)
  PORT                       Bind port (default: $PORT)
  HEALTH_HOST                Host used for local health checks (default: $HEALTH_HOST)
  DRAIN_URL                  Local drain endpoint (default: $DRAIN_URL)
  WORKER_STATE_URL           Local worker-state endpoint (default: $WORKER_STATE_URL)
  LOG_DIR                    Server log dir (default: $LOG_DIR)
  TURN_ENV_FILE              Optional WebRTC/TURN env file (default: $TURN_ENV_FILE)
  WEBRTC_RELAY_ENABLED       Use scripts/run_webrtc_relay_api_server.sh (default: $WEBRTC_RELAY_ENABLED)
  WEBRTC_TURN_AUTOSTART      Start/stop local coturn via scripts/run_turnserver_tcp_relay.sh (default: $WEBRTC_TURN_AUTOSTART)
  TURN_LOG_FILE              Coturn log file when autostart is enabled (default: $TURN_LOG_FILE)
  TURN_PID_FILE              Coturn pid file when autostart is enabled (default: $TURN_PID_FILE)
  STARTUP_TIMEOUT_SECONDS    Health wait timeout (default: $STARTUP_TIMEOUT_SECONDS)
  POLL_INTERVAL_SECONDS      Health wait poll interval (default: $POLL_INTERVAL_SECONDS)
  DRAIN_TIMEOUT_SECONDS      Drain wait timeout before stop (default: $DRAIN_TIMEOUT_SECONDS)

Examples:
  $SCRIPT_NAME start
  PROFILE=throughput_record PORT=8010 $SCRIPT_NAME restart
  $SCRIPT_NAME status
  $SCRIPT_NAME logs
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

env_enabled() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

load_turn_env() {
  if (( TURN_ENV_LOADED )); then
    return 0
  fi
  TURN_ENV_LOADED=1
  if ! env_enabled "$VAST_SERVER_CTL_LOAD_TURN_ENV"; then
    return 0
  fi
  if [[ ! -f "$TURN_ENV_FILE" ]]; then
    return 0
  fi

  set -a
  # shellcheck disable=SC1090
  source "$TURN_ENV_FILE"
  set +a
}

resolve_pid() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    return 1
  fi
  printf '%s\n' "$pid"
}

cleanup_stale_pid() {
  if [[ -f "$PID_FILE" ]] && ! resolve_pid >/dev/null 2>&1; then
    rm -f "$PID_FILE"
  fi
}

resolve_turn_pid() {
  if [[ ! -f "$TURN_PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(tr -d '[:space:]' < "$TURN_PID_FILE")"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    return 1
  fi
  printf '%s\n' "$pid"
}

cleanup_stale_turn_pid() {
  if [[ -f "$TURN_PID_FILE" ]] && ! resolve_turn_pid >/dev/null 2>&1; then
    rm -f "$TURN_PID_FILE"
  fi
}

check_health() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "$HEALTH_URL" >/dev/null 2>&1
    return $?
  fi

  python3 - <<PY >/dev/null 2>&1
import sys
import urllib.request

try:
    with urllib.request.urlopen(${HEALTH_URL@Q}, timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

wait_for_health() {
  local start_ts now elapsed pid
  start_ts="$(date +%s)"
  while true; do
    if check_health; then
      elapsed="$(( $(date +%s) - start_ts ))"
      log "Health check passed at $HEALTH_URL after $(format_duration "$elapsed")"
      return 0
    fi

    if pid="$(resolve_pid 2>/dev/null)"; then
      :
    else
      log "Server process exited before health check passed"
      if [[ -f "$LOG_FILE" ]]; then
        printf '\n[%s] Last log lines from %s:\n' "$SCRIPT_NAME" "$LOG_FILE" >&2
        tail -n 40 "$LOG_FILE" >&2 || true
      fi
      return 1
    fi

    now="$(date +%s)"
    elapsed="$((now - start_ts))"
    if (( elapsed >= STARTUP_TIMEOUT_SECONDS )); then
      log "Timed out waiting for health after ${elapsed}s"
      if [[ -f "$LOG_FILE" ]]; then
        printf '\n[%s] Last log lines from %s:\n' "$SCRIPT_NAME" "$LOG_FILE" >&2
        tail -n 40 "$LOG_FILE" >&2 || true
      fi
      return 1
    fi
    sleep "$POLL_INTERVAL_SECONDS"
  done
}

request_drain() {
  if ! command -v curl >/dev/null 2>&1; then
    return 1
  fi
  curl -fsS -X POST "$DRAIN_URL" >/dev/null 2>&1
}

worker_is_idle() {
  python3 - <<PY >/dev/null 2>&1
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(${WORKER_STATE_URL@Q}, timeout=2) as response:
        payload = json.load(response)
except Exception:
    sys.exit(1)

metrics = payload.get("metrics") or {}
active_requests = int(metrics.get("active_requests", 0) or 0)
active_sessions = int(metrics.get("active_sessions_local", 0) or 0)
queue_depth = int(metrics.get("queue_depth", 0) or 0)
sys.exit(0 if active_requests <= 0 and active_sessions <= 0 and queue_depth <= 0 else 1)
PY
}

wait_for_drain() {
  local start_ts now elapsed
  start_ts="$(date +%s)"
  while true; do
    if worker_is_idle; then
      log "Drain complete"
      return 0
    fi

    now="$(date +%s)"
    elapsed="$((now - start_ts))"
    if (( elapsed >= DRAIN_TIMEOUT_SECONDS )); then
      log "Timed out waiting for drain after ${elapsed}s"
      return 1
    fi
    sleep "$POLL_INTERVAL_SECONDS"
  done
}

start_server() {
  local start_ts elapsed
  start_ts="$(date +%s)"
  cleanup_stale_pid
  cleanup_stale_turn_pid
  mkdir -p "$LOG_DIR"

  if resolve_pid >/dev/null 2>&1; then
    local pid
    pid="$(resolve_pid)"
    log "Server already running with pid=$pid"
    check_health && log "Health already OK at $HEALTH_URL" || log "Process is running but health is not ready yet"
    return 0
  fi

  [[ -x "$VENV_PATH/bin/python" ]] || die "Venv python not found at $VENV_PATH/bin/python"
  [[ -f "$REPO_ROOT/api_server.py" ]] || die "api_server.py not found under $REPO_ROOT"

  local launcher="$REPO_ROOT/scripts/run_trt_stagewise_server.sh"
  if env_enabled "$WEBRTC_RELAY_ENABLED"; then
    launcher="$REPO_ROOT/scripts/run_webrtc_relay_api_server.sh"
    start_turnserver
  fi

  log "Starting MuseTalk server"
  log "repo=$REPO_ROOT"
  log "venv=$VENV_PATH"
  log "profile=$PROFILE host=$HOST port=$PORT"
  log "launcher=$launcher"
  log "webrtc_relay_enabled=$WEBRTC_RELAY_ENABLED turn_autostart=$WEBRTC_TURN_AUTOSTART turn_env=$TURN_ENV_FILE"
  log "log_file=$LOG_FILE"

  setsid nohup bash "$launcher" \
    --profile "$PROFILE" \
    --host "$HOST" \
    --port "$PORT" \
    --venv-path "$VENV_PATH" \
    --repo-root "$REPO_ROOT" \
    >>"$LOG_FILE" 2>&1 &

  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  log "Spawned pid=$pid"
  wait_for_health
  elapsed="$(( $(date +%s) - start_ts ))"
  log "Start command completed in $(format_duration "$elapsed")"
}

start_turnserver() {
  if ! env_enabled "$WEBRTC_TURN_AUTOSTART"; then
    return 0
  fi

  cleanup_stale_turn_pid
  mkdir -p "$LOG_DIR"

  if resolve_turn_pid >/dev/null 2>&1; then
    local pid
    pid="$(resolve_turn_pid)"
    log "TURN server already running with pid=$pid"
    return 0
  fi

  if ! command -v turnserver >/dev/null 2>&1; then
    die "WEBRTC_TURN_AUTOSTART=1 but turnserver is not installed. Install coturn or disable autostart and provide external WEBRTC_TURN_URLS."
  fi

  if [[ -z "${TURN_PASS:-${WEBRTC_TURN_PASS:-}}" ]]; then
    die "WEBRTC_TURN_AUTOSTART=1 requires TURN_PASS or WEBRTC_TURN_PASS in the environment or $TURN_ENV_FILE"
  fi

  log "Starting local TURN server"
  log "turn_log_file=$TURN_LOG_FILE"
  setsid nohup bash "$REPO_ROOT/scripts/run_turnserver_tcp_relay.sh" \
    >>"$TURN_LOG_FILE" 2>&1 &

  local pid=$!
  printf '%s\n' "$pid" > "$TURN_PID_FILE"
  sleep 1
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    rm -f "$TURN_PID_FILE"
    if [[ -f "$TURN_LOG_FILE" ]]; then
      printf '\n[%s] Last TURN log lines from %s:\n' "$SCRIPT_NAME" "$TURN_LOG_FILE" >&2
      tail -n 40 "$TURN_LOG_FILE" >&2 || true
    fi
    die "TURN server exited during startup"
  fi
  log "TURN server spawned pid=$pid"
}

stop_server() {
  cleanup_stale_pid
  if ! resolve_pid >/dev/null 2>&1; then
    log "Server is not running"
    stop_turnserver
    return 0
  fi

  local pid
  pid="$(resolve_pid)"

  if check_health; then
    if request_drain; then
      log "Requested drain via $DRAIN_URL"
      wait_for_drain || true
    else
      log "Drain endpoint unavailable; falling back to signal-based shutdown"
    fi
  fi

  log "Stopping pid=$pid"
  kill "$pid" >/dev/null 2>&1 || true

  local tries="$(( DRAIN_TIMEOUT_SECONDS > 15 ? DRAIN_TIMEOUT_SECONDS : 15 ))"
  while (( tries > 0 )); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -f "$PID_FILE"
      log "Server stopped"
      stop_turnserver
      return 0
    fi
    sleep 1
    tries="$((tries - 1))"
  done

  log "Process did not exit in time; sending SIGKILL"
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$PID_FILE"
  stop_turnserver
}

stop_turnserver() {
  cleanup_stale_turn_pid
  if ! env_enabled "$WEBRTC_TURN_AUTOSTART" && [[ ! -f "$TURN_PID_FILE" ]]; then
    return 0
  fi
  if ! resolve_turn_pid >/dev/null 2>&1; then
    return 0
  fi

  local pid
  pid="$(resolve_turn_pid)"
  log "Stopping TURN server pid=$pid"
  kill "$pid" >/dev/null 2>&1 || true

  local tries=10
  while (( tries > 0 )); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -f "$TURN_PID_FILE"
      log "TURN server stopped"
      return 0
    fi
    sleep 1
    tries="$((tries - 1))"
  done

  log "TURN server did not exit in time; sending SIGKILL"
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$TURN_PID_FILE"
}

status_server() {
  cleanup_stale_pid
  cleanup_stale_turn_pid
  if resolve_pid >/dev/null 2>&1; then
    local pid
    pid="$(resolve_pid)"
    if check_health; then
      log "running pid=$pid health=ok url=$HEALTH_URL"
    else
      log "running pid=$pid health=starting url=$HEALTH_URL"
    fi
  else
    log "stopped"
  fi

  if resolve_turn_pid >/dev/null 2>&1; then
    local turn_pid
    turn_pid="$(resolve_turn_pid)"
    log "turnserver=running pid=$turn_pid"
  elif env_enabled "$WEBRTC_TURN_AUTOSTART"; then
    log "turnserver=stopped"
  fi
}

logs_server() {
  mkdir -p "$LOG_DIR"
  if [[ ! -f "$LOG_FILE" ]]; then
    log "Log file not found: $LOG_FILE"
    return 0
  fi
  tail -n 200 "$LOG_FILE"
}

main() {
  load_turn_env
  WEBRTC_RELAY_ENABLED="${WEBRTC_RELAY_ENABLED:-0}"
  WEBRTC_TURN_AUTOSTART="${WEBRTC_TURN_AUTOSTART:-0}"
  TURN_LOG_FILE="${TURN_LOG_FILE:-$LOG_DIR/turnserver.log}"
  TURN_PID_FILE="${TURN_PID_FILE:-$LOG_DIR/turnserver.pid}"

  local command="${1:-}"
  case "$command" in
    start)
      start_server
      ;;
    stop)
      stop_server
      ;;
    restart)
      stop_server
      start_server
      ;;
    status)
      status_server
      ;;
    logs)
      logs_server
      ;;
    --help|-h|help|"")
      usage
      ;;
    *)
      die "Unknown command: $command"
      ;;
  esac
}

main "$@"
