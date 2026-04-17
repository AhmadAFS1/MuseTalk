#!/usr/bin/env bash

if [[ -n "${MUSETALK_STEP_LOGGING_LOADED:-}" ]]; then
  return 0
fi
MUSETALK_STEP_LOGGING_LOADED=1

STEP_LOG_SCRIPT_NAME="${STEP_LOG_SCRIPT_NAME:-script}"
STEP_LOG_FILE="${STEP_LOG_FILE:-}"
STEP_LOG_SUMMARY_PRINTED=0
declare -ag STEP_LOG_SUMMARY=()

step_format_duration() {
  local total_seconds="${1:-0}"
  local minutes=$((total_seconds / 60))
  local seconds=$((total_seconds % 60))

  if (( minutes > 0 )); then
    printf '%dm%02ds' "$minutes" "$seconds"
  else
    printf '%ss' "$seconds"
  fi
}

step_log_emit() {
  local level="$1"
  shift

  local timestamp
  timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"

  printf '[%s][%s][%s] %s\n' "$STEP_LOG_SCRIPT_NAME" "$level" "$timestamp" "$*"
  if [[ -n "$STEP_LOG_FILE" ]]; then
    printf '[%s][%s][%s] %s\n' "$STEP_LOG_SCRIPT_NAME" "$level" "$timestamp" "$*" >>"$STEP_LOG_FILE"
  fi
}

step_logging_init() {
  local script_name="$1"
  local log_dir="${2:-}"

  STEP_LOG_SCRIPT_NAME="$script_name"
  if [[ -z "$STEP_LOG_FILE" && -n "$log_dir" ]]; then
    mkdir -p "$log_dir"
    STEP_LOG_FILE="$log_dir/${script_name%.sh}-$(date +%Y%m%dT%H%M%S).log"
  fi

  step_log_emit INFO "Initialized step logging"
  if [[ -n "$STEP_LOG_FILE" ]]; then
    step_log_emit INFO "Log file: $STEP_LOG_FILE"
  fi
}

step_record_result() {
  local label="$1"
  local elapsed_seconds="$2"
  local status="$3"

  STEP_LOG_SUMMARY+=("${status}"$'\t'"${elapsed_seconds}"$'\t'"${label}")
}

run_step() {
  local label="$1"
  shift

  local start_ts
  local end_ts
  local elapsed_seconds
  local status=0

  start_ts="$(date +%s)"
  step_log_emit STEP "START: $label"
  if "$@"; then
    status=0
  else
    status=$?
  fi
  end_ts="$(date +%s)"
  elapsed_seconds=$((end_ts - start_ts))

  if [[ $status -eq 0 ]]; then
    step_record_result "$label" "$elapsed_seconds" "OK"
    step_log_emit STEP "DONE: $label ($(step_format_duration "$elapsed_seconds"))"
    return 0
  fi

  step_record_result "$label" "$elapsed_seconds" "FAIL"
  step_log_emit ERROR "FAILED: $label ($(step_format_duration "$elapsed_seconds"), exit=$status)"
  return "$status"
}

print_step_summary() {
  local entry
  local status
  local elapsed_seconds
  local label

  if [[ ${#STEP_LOG_SUMMARY[@]} -eq 0 ]]; then
    step_log_emit INFO "No timed steps were recorded"
    return
  fi

  step_log_emit INFO "Step summary:"
  for entry in "${STEP_LOG_SUMMARY[@]}"; do
    IFS=$'\t' read -r status elapsed_seconds label <<<"$entry"
    step_log_emit INFO "  [$status] $label ($(step_format_duration "$elapsed_seconds"))"
  done
}

step_logging_on_exit() {
  local exit_code="${1:-0}"

  if [[ $STEP_LOG_SUMMARY_PRINTED -eq 0 ]]; then
    print_step_summary
    STEP_LOG_SUMMARY_PRINTED=1
  fi

  if [[ $exit_code -eq 0 ]]; then
    step_log_emit INFO "Exit status: success"
  else
    step_log_emit ERROR "Exit status: $exit_code"
  fi
}
