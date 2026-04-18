#!/usr/bin/env bash

if [[ -n "${MUSETALK_STEP_LOGGING_LOADED:-}" ]]; then
  return 0
fi
MUSETALK_STEP_LOGGING_LOADED=1

STEP_LOG_SCRIPT_NAME="${STEP_LOG_SCRIPT_NAME:-script}"
STEP_LOG_FILE="${STEP_LOG_FILE:-}"
STEP_LOG_SUMMARY_PRINTED=0
STEP_LOG_HOTSPOT_LIMIT="${STEP_LOG_HOTSPOT_LIMIT:-5}"
STEP_LOG_HOTSPOT_MIN_SECONDS="${STEP_LOG_HOTSPOT_MIN_SECONDS:-10}"
declare -ag STEP_LOG_SUMMARY=()
declare -ag STEP_LOG_PHASE_SUMMARY=()
STEP_LOG_CURRENT_PHASE="${STEP_LOG_CURRENT_PHASE:-}"

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

step_record_phase_result() {
  local label="$1"
  local description="$2"
  local elapsed_seconds="$3"
  local status="$4"

  STEP_LOG_PHASE_SUMMARY+=("${status}"$'\t'"${elapsed_seconds}"$'\t'"${label}"$'\t'"${description}")
}

run_step() {
  local label="$1"
  shift

  local qualified_label="$label"
  local start_ts
  local end_ts
  local elapsed_seconds
  local status=0

  if [[ -n "$STEP_LOG_CURRENT_PHASE" ]]; then
    qualified_label="$STEP_LOG_CURRENT_PHASE :: $label"
  fi

  start_ts="$(date +%s)"
  step_log_emit STEP "START: $qualified_label"
  if "$@"; then
    status=0
  else
    status=$?
  fi
  end_ts="$(date +%s)"
  elapsed_seconds=$((end_ts - start_ts))

  if [[ $status -eq 0 ]]; then
    step_record_result "$qualified_label" "$elapsed_seconds" "OK"
    step_log_emit STEP "DONE: $qualified_label ($(step_format_duration "$elapsed_seconds"))"
    return 0
  fi

  step_record_result "$qualified_label" "$elapsed_seconds" "FAIL"
  step_log_emit ERROR "FAILED: $qualified_label ($(step_format_duration "$elapsed_seconds"), exit=$status)"
  return "$status"
}

run_phase() {
  local phase_id="$1"
  local title="$2"
  local description="$3"
  shift 3

  local phase_label="${phase_id}: ${title}"
  local previous_phase="${STEP_LOG_CURRENT_PHASE:-}"
  local start_ts
  local end_ts
  local elapsed_seconds
  local status=0

  start_ts="$(date +%s)"
  step_log_emit PHASE "START: $phase_label"
  if [[ -n "$description" ]]; then
    step_log_emit PHASE "DETAIL: $description"
  fi

  STEP_LOG_CURRENT_PHASE="$phase_label"
  if "$@"; then
    status=0
  else
    status=$?
  fi
  STEP_LOG_CURRENT_PHASE="$previous_phase"

  end_ts="$(date +%s)"
  elapsed_seconds=$((end_ts - start_ts))

  if [[ $status -eq 0 ]]; then
    step_record_phase_result "$phase_label" "$description" "$elapsed_seconds" "OK"
    step_log_emit PHASE "DONE: $phase_label ($(step_format_duration "$elapsed_seconds"))"
    return 0
  fi

  step_record_phase_result "$phase_label" "$description" "$elapsed_seconds" "FAIL"
  step_log_emit ERROR "FAILED: $phase_label ($(step_format_duration "$elapsed_seconds"), exit=$status)"
  return "$status"
}

print_phase_summary() {
  local entry
  local status
  local elapsed_seconds
  local label
  local description

  if [[ ${#STEP_LOG_PHASE_SUMMARY[@]} -eq 0 ]]; then
    return
  fi

  step_log_emit INFO "Phase summary:"
  for entry in "${STEP_LOG_PHASE_SUMMARY[@]}"; do
    IFS=$'\t' read -r status elapsed_seconds label description <<<"$entry"
    if [[ -n "$description" ]]; then
      step_log_emit INFO "  [$status] $label ($(step_format_duration "$elapsed_seconds")) - $description"
    else
      step_log_emit INFO "  [$status] $label ($(step_format_duration "$elapsed_seconds"))"
    fi
  done
}

step_print_sorted_hotspots() {
  local heading="$1"
  local entry_kind="$2"
  local array_name="$3"
  local -n entries_ref="$array_name"
  local count=0
  local entry
  local status
  local elapsed_seconds
  local label
  local description

  if [[ ${#entries_ref[@]} -eq 0 ]]; then
    return
  fi

  while IFS= read -r entry; do
    [[ -n "$entry" ]] || continue

    if [[ "$entry_kind" == "phase" ]]; then
      IFS=$'\t' read -r status elapsed_seconds label description <<<"$entry"
    else
      IFS=$'\t' read -r status elapsed_seconds label <<<"$entry"
      description=""
    fi

    (( elapsed_seconds >= STEP_LOG_HOTSPOT_MIN_SECONDS )) || continue

    if (( count == 0 )); then
      step_log_emit INFO "$heading:"
    fi

    if [[ -n "$description" ]]; then
      step_log_emit INFO "  [$status] $label ($(step_format_duration "$elapsed_seconds")) - $description"
    else
      step_log_emit INFO "  [$status] $label ($(step_format_duration "$elapsed_seconds"))"
    fi

    count=$((count + 1))
    if (( count >= STEP_LOG_HOTSPOT_LIMIT )); then
      break
    fi
  done < <(printf '%s\n' "${entries_ref[@]}" | sort -t $'\t' -k2,2nr)
}

print_hotspot_summary() {
  step_print_sorted_hotspots "Slowest phases" "phase" "STEP_LOG_PHASE_SUMMARY"
  step_print_sorted_hotspots "Slowest steps" "step" "STEP_LOG_SUMMARY"
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
    print_phase_summary
    print_hotspot_summary
    print_step_summary
    STEP_LOG_SUMMARY_PRINTED=1
  fi

  if [[ $exit_code -eq 0 ]]; then
    step_log_emit INFO "Exit status: success"
  else
    step_log_emit ERROR "Exit status: $exit_code"
  fi
}
