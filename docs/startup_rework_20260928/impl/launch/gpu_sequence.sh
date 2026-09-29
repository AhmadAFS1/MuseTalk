#!/usr/bin/env bash
# GPU validation sequence for the launch chain (component D, startup rework 2026-09-28).
#
# Written CPU-only; NOT executed by its author. Run it on the RTX 4070 SUPER box when the GPU
# is free (box_guard waits up to 60 min for the user's server / other tenants, then exits 75):
#
#   bash docs/startup_rework_20260928/impl/launch/gpu_sequence.sh --list
#   bash docs/startup_rework_20260928/impl/launch/gpu_sequence.sh all
#   bash docs/startup_rework_20260928/impl/launch/gpu_sequence.sh boot_fast verify_negative
#
# Every GPU step runs as ONE command under
#   scripts/box_guard.sh run --min-avail-gb N --wait-min 60 --label startup_<step> -- <cmd>
# and prints exactly one "PASS <step>" / "FAIL <step>" / "BLOCKED <step>" line. Results are JSON
# files next to this script (<step>.json) plus gpu_sequence_summary.json. Boot steps use their own
# port (831x), LOG_DIR and resolved/launch-state files under this directory, so they never touch
# the live server's :8000 files or the shared .runtime/musetalk_resolved.* of the main checkout.
# Every boot step stops its server before returning (trap), so the lease is never held by a
# long-lived server.
#
#   step                     GPU  box_guard min-avail  expected wall   expected peak host RAM
#   cpu_tests                no   -                    ~30 s           <0.5 GB
#   cpu_print_env            no   -                    ~5 s            <0.2 GB (resolver is stdlib)
#   cpu_validate_only        no   -                    ~5 s            <0.3 GB (+ native VP8 preflight if enabled)
#   engines_unet_ts          yes  12 GB                2-5 min         ~9.5 GB (adopt + validate the 2.2 GB .ts)
#   engines_taesd_trt        yes  6 GB                 0.5-1.5 min     ~3-4 GB (build ~15 s + probe)
#   engines_unet_stagewise   yes  8 GB                 1-3 min         ~4-6 GB (adopt + validate bs16 set)
#   boot_fast                yes  14 GB                2-3 min         ~10.5 GB transient (.ts load), ~6 GB steady
#   boot_fast300             yes  14 GB                2-3 min         as boot_fast (all fast300 levers are off today)
#   boot_fast300_candidate   yes  10 GB                2-3 min         ~6 GB (stagewise bs16 + TAESD TRT, no .ts)
#   verify_negative          yes  8 GB                 1-2 min         ~5 GB (eager UNet, TAESD TRT forced to fall back)
#   legacy_rollback          yes  14 GB                1.5-3 min       ~10.5 GB (old int8 chain + sm89 profile env)
#   summary                  no   -                    <1 s            -
# Total with every step: ~15-25 min of GPU lease, run back to back (steps release the lease between them).

set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/workspace/MuseTalk-perf300}"
VENV_PATH="${VENV_PATH:-/workspace/.venvs/musetalk_trt_stagewise}"
PY="$VENV_PATH/bin/python"
BOX_GUARD="${BOX_GUARD:-$REPO/scripts/box_guard.sh}"
OUT="${OUT:-$HERE}"
WAIT_MIN="${WAIT_MIN:-60}"
SCRATCH="${SCRATCH:-$OUT/scratch}"

ALL_STEPS=(cpu_tests cpu_print_env cpu_validate_only engines_unet_ts engines_taesd_trt engines_unet_stagewise
           boot_fast boot_fast300 boot_fast300_candidate verify_negative legacy_rollback summary)

log() { printf '[gpu_sequence %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

write_result() {
  # write_result STEP RESULT RC STARTED_TS [KEY=VALUE ...]
  local step="$1" result="$2" rc="$3" started="$4"
  shift 4
  "$PY" -I -B - "$OUT/$step.json" "$step" "$result" "$rc" "$started" "$@" <<'PY'
import json, sys, time
path, step, result, rc, started = sys.argv[1:6]
extra = {}
for item in sys.argv[6:]:
    key, _, value = item.partition("=")
    try:
        extra[key] = json.loads(value)
    except ValueError:
        extra[key] = value
record = {"step": step, "result": result, "rc": int(rc),
          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(started))),
          "elapsed_s": round(time.time() - float(started), 1), **extra}
json.dump(record, open(path, "w"), indent=1, sort_keys=True)
PY
}

verdict() {
  # verdict STEP RC STARTED [extra...]: 0 -> PASS, 75 -> BLOCKED (lease/GPU wait timeout), else FAIL
  local step="$1" rc="$2" started="$3"
  shift 3
  local result=FAIL
  if (( rc == 0 )); then
    result=PASS
  elif (( rc == 75 )); then
    result=BLOCKED
  fi
  write_result "$step" "$result" "$rc" "$started" "$@"
  printf '%s %s (exit %s, %ss) -> %s\n' "$result" "$step" "$rc" "$(( $(date +%s) - started ))" "$OUT/$step.json"
  return 0
}

guarded() {
  # guarded STEP MIN_AVAIL_GB CMD... : box_guard wrapper
  local step="$1" min_gb="$2"
  shift 2
  "$BOX_GUARD" run --min-avail-gb "$min_gb" --wait-min "$WAIT_MIN" --label "startup_$step" -- "$@"
}

# Environment for validation boots: no control plane, no relay/TURN, no S3, no network downloads.
boot_env() {
  local name="$1" port="$2"
  mkdir -p "$OUT/$name.d/logs"
  export REPO_ROOT="$REPO" VENV_PATH PORT="$port" HOST=127.0.0.1 HEALTH_HOST=127.0.0.1
  export LOG_DIR="$OUT/$name.d/logs"
  export MUSETALK_RESOLVED_ENV_FILE="$OUT/$name.d/musetalk_resolved.env"
  export MUSETALK_RESOLVED_REPORT_FILE="$OUT/$name.d/musetalk_resolved.json"
  export MUSETALK_LAUNCH_STATE_FILE="$OUT/$name.d/musetalk_launch.json"
  export MUSETALK_ENV_OVERRIDES_FILE="${BOOT_OVERRIDES:-$OUT/$name.d/none.env}"
  export LINGUA_CONTROL_PLANE_ENV_FILE="$OUT/$name.d/no-control-plane.env"
  export VAST_SERVER_CTL_LOAD_TURN_ENV=0 WEBRTC_RELAY_ENABLED=0 WEBRTC_TURN_AUTOSTART=0
  export AVATAR_S3_ENABLED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-900}" MUSETALK_VERIFY_TIMEOUT_SECONDS=90
  unset LINGUA_WORKER_TOKEN LINGUA_CONTROL_PLANE_BASE_URL LINGUA_WORKER_REGISTER_URL LINGUA_WORKER_HEARTBEAT_URL
}

# Internal: one boot = start (health + strict verify) -> status -> /health snapshot -> stop.
# Runs INSIDE box_guard. Exit 0 = the expectation of this boot held.
internal_boot() {
  local name="$1" port="$2" expect="$3"   # expect: pass | verify_fail | skip
  boot_env "$name" "$port"
  local dir="$OUT/$name.d" ctl="$REPO/scripts/vast_server_ctl.sh" rc=0
  local stopped=0
  stop_boot() {
    (( stopped )) && return 0
    stopped=1
    bash "$ctl" stop > "$dir/ctl_stop.out" 2>&1 || true
  }
  trap stop_boot EXIT
  log "boot $name on :$port (expect=$expect)"
  bash "$ctl" start > "$dir/ctl_start.out" 2>&1 || rc=$?
  bash "$ctl" status > "$dir/ctl_status.out" 2>&1 || true
  curl -fsS "http://127.0.0.1:$port/health" > "$dir/health.json" 2>/dev/null || true
  grep -E 'backend active|backend: PyTorch|TAESD TRT|Loaded TAESD|stagewise|Eager UNet released' \
    "$LOG_DIR/api_server_${port}.log" > "$dir/backend_lines.txt" 2>/dev/null || true
  grep -E 'Health check passed|Start command completed' "$dir/ctl_start.out" > "$dir/timing.txt" || true
  stop_boot
  local verify_line
  verify_line="$(cat "$LOG_DIR/api_server_${port}.verify" 2>/dev/null || echo 'missing')"
  printf '%s\n' "$verify_line" > "$dir/verify.txt"
  case "$expect" in
    pass)
      (( rc == 0 )) && [[ "$verify_line" == *result=PASS* ]]
      ;;
    verify_fail)
      # strict verification must refuse the silent fallback and stop the server
      (( rc != 0 )) && [[ "$verify_line" == *result=FAIL* ]] \
        && ! curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1
      ;;
    skip)
      (( rc == 0 )) && [[ "$verify_line" == *result=SKIP* ]]
      ;;
  esac
}

run_boot_step() {
  # run_boot_step STEP PORT MIN_GB EXPECT [ENV=VALUE ...]
  local step="$1" port="$2" min_gb="$3" expect="$4"
  shift 4
  local started rc=0
  started="$(date +%s)"
  mkdir -p "$OUT/$step.d"
  # __boot runs box_guard around __boot_inner (start -> verify -> status -> stop)
  env "$@" bash "$0" __boot "$step" "$port" "$expect" > "$OUT/$step.d/step.out" 2>&1 || rc=$?
  verdict "$step" "$rc" "$started" \
    "expect=$expect" "port=$port" \
    "verify=$(tr -d '\n' < "$OUT/$step.d/verify.txt" 2>/dev/null | "$PY" -I -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
    "launch_state=$OUT/$step.d/musetalk_launch.json" \
    "resolved_report=$OUT/$step.d/musetalk_resolved.json" \
    "server_log=$OUT/$step.d/logs/api_server_${port}.log"
}

step_cpu_tests() {
  local started rc=0
  started="$(date +%s)"
  mkdir -p "$SCRATCH"
  TMPDIR="$SCRATCH" bash "$REPO/scripts/test_startup_scripts.sh" > "$OUT/cpu_tests.out" 2>&1 || rc=$?
  verdict cpu_tests "$rc" "$started" "tail=$(tail -n 1 "$OUT/cpu_tests.out" | "$PY" -I -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')"
}

step_cpu_print_env() {
  local started rc=0 recipe
  started="$(date +%s)"
  for recipe in fast fast300; do
    MUSETALK_RECIPE="$recipe" MUSETALK_ENV_OVERRIDES_FILE="$OUT/none.env" \
      bash "$REPO/scripts/run_musetalk_server.sh" --venv-path "$VENV_PATH" --repo-root "$REPO" --port 8310 \
      --print-env > "$OUT/print_env_$recipe.txt" 2> "$OUT/print_env_$recipe.err" || rc=$?
  done
  if (( rc == 0 )); then
    grep -q "^MUSETALK_VAE_BACKEND='taesd'" "$OUT/print_env_fast.txt" || rc=1
    grep -q "^MUSETALK_RECIPE='fast300'" "$OUT/print_env_fast300.txt" || rc=1
  fi
  verdict cpu_print_env "$rc" "$started" \
    "unet_fast=$(sed -nE "s/^MUSETALK_UNET_BACKEND='([^']*)'.*/\1/p" "$OUT/print_env_fast.txt" | "$PY" -I -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')"
}

step_cpu_validate_only() {
  local started rc=0
  started="$(date +%s)"
  MUSETALK_ENV_OVERRIDES_FILE="$OUT/none.env" MUSETALK_RESOLVED_ENV_FILE="$OUT/validate/musetalk_resolved.env" \
    MUSETALK_RESOLVED_REPORT_FILE="$OUT/validate/musetalk_resolved.json" \
    bash "$REPO/scripts/run_musetalk_server.sh" --venv-path "$VENV_PATH" --repo-root "$REPO" --port 8310 \
    --validate-only > "$OUT/validate_only.out" 2>&1 || rc=$?
  verdict cpu_validate_only "$rc" "$started" "report=$OUT/validate/musetalk_resolved.validate.json"
}

step_engine() {
  # step_engine STEP MIN_GB KIND MODE [--batch N]
  local step="$1" min_gb="$2" kind="$3" mode="$4" started rc=0
  shift 4
  started="$(date +%s)"
  (
    cd "$REPO"
    guarded "$step" "$min_gb" "$PY" scripts/unet_engine_store.py ensure --kind "$kind" "$@" --provision "$mode"
  ) > "$OUT/$step.out" 2>&1 || rc=$?
  (cd "$REPO" && "$PY" scripts/unet_engine_store.py list > "$OUT/${step}_list.json" 2>/dev/null) || true
  verdict "$step" "$rc" "$started" "kind=$kind" "provision=$mode"
}

step_summary() {
  local started
  started="$(date +%s)"
  "$PY" -I -B - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
steps = {}
for path in sorted(glob.glob(os.path.join(out, "*.json"))):
    name = os.path.basename(path)[:-5]
    if name in ("gpu_sequence_summary",) or name.endswith("_list"):
        continue
    try:
        record = json.load(open(path))
    except Exception:
        continue
    if isinstance(record, dict) and "result" in record:
        steps[name] = {"result": record["result"], "rc": record.get("rc"), "elapsed_s": record.get("elapsed_s")}
summary = {"steps": steps,
           "all_pass": bool(steps) and all(v["result"] == "PASS" for v in steps.values())}
json.dump(summary, open(os.path.join(out, "gpu_sequence_summary.json"), "w"), indent=1, sort_keys=True)
for name, v in steps.items():
    print(f"{v['result']:8s} {name}")
PY
  printf 'PASS summary -> %s\n' "$OUT/gpu_sequence_summary.json"
}

run_step() {
  case "$1" in
    cpu_tests) step_cpu_tests ;;
    cpu_print_env) step_cpu_print_env ;;
    cpu_validate_only) step_cpu_validate_only ;;
    engines_unet_ts) step_engine engines_unet_ts 12 unet_ts "${UNET_TS_PROVISION:-adopt}" ;;
    engines_taesd_trt) step_engine engines_taesd_trt 6 taesd_trt "${TAESD_TRT_PROVISION:-auto}" ;;
    engines_unet_stagewise) step_engine engines_unet_stagewise 8 unet_stagewise "${STAGEWISE_PROVISION:-adopt}" ;;
    boot_fast)
      run_boot_step boot_fast 8310 14 pass MUSETALK_RECIPE=fast
      ;;
    boot_fast300)
      run_boot_step boot_fast300 8311 14 pass MUSETALK_RECIPE=fast300
      ;;
    boot_fast300_candidate)
      # The gated fast300 engine levers switched on explicitly through an overrides file, to prove
      # the plumbing (expectations vae=taesd_trt unet=trt_stagewise, strict verification).
      mkdir -p "$OUT/boot_fast300_candidate.d"
      printf '%s\n' MUSETALK_UNET_BACKEND=trt_stagewise MUSETALK_UNET_STAGEWISE_BATCH=16 \
        HLS_SCHEDULER_FIXED_BATCH_SIZES=16 MUSETALK_TAESD_BACKEND=trt MUSETALK_TAESD_TRT_BUILD=0 \
        MUSETALK_TAESD_TRT_STRICT=1 > "$OUT/boot_fast300_candidate.d/candidate_overrides.env"
      run_boot_step boot_fast300_candidate 8312 10 pass MUSETALK_RECIPE=fast300 \
        BOOT_OVERRIDES="$OUT/boot_fast300_candidate.d/candidate_overrides.env"
      ;;
    verify_negative)
      # TAESD TRT requested but impossible (empty engine dir, no build, STRICT=0): the server silently
      # serves compiled TAESD; strict verification must catch it and stop the server.
      mkdir -p "$OUT/verify_negative.d/empty_taesd_trt"
      run_boot_step verify_negative 8313 8 verify_fail MUSETALK_RECIPE=fast MUSETALK_UNET_MODE=eager \
        MUSETALK_TAESD_BACKEND=trt MUSETALK_TAESD_TRT_BUILD=0 MUSETALK_TAESD_TRT_STRICT=0 \
        MUSETALK_TAESD_TRT_DIR="$OUT/verify_negative.d/empty_taesd_trt"
      ;;
    legacy_rollback)
      run_boot_step legacy_rollback 8314 14 skip MUSETALK_RECIPE=legacy_int8 \
        MUSETALK_TRT_PROFILE_ENV_LOAD=1 MUSETALK_TRT_PROFILE_ENV_FILE="$REPO/.runtime/musetalk_trt_local_sm89.env"
      ;;
    summary) step_summary ;;
    *) log "unknown step $1"; return 2 ;;
  esac
}

main() {
  if [[ "${1:-}" == "--list" ]]; then
    printf '%s\n' "${ALL_STEPS[@]}"
    return 0
  fi
  mkdir -p "$OUT"
  : > "$OUT/none.env"
  if [[ "${1:-}" == "__boot" ]]; then
    # __boot STEP PORT EXPECT: the guarded boot (box_guard -> internal_boot)
    local step="$2" port="$3" expect="$4"
    local min_gb
    case "$step" in
      boot_fast300_candidate) min_gb=10 ;;
      verify_negative) min_gb=8 ;;
      *) min_gb=14 ;;
    esac
    local rc=0
    guarded "$step" "$min_gb" bash "$0" __boot_inner "$step" "$port" "$expect" || rc=$?
    # Safety net outside the lease: if the watchdog killed the inner step, its trap never ran.
    boot_env "$step" "$port"
    bash "$REPO/scripts/vast_server_ctl.sh" stop > "$OUT/$step.d/ctl_stop_safety.out" 2>&1 || true
    return "$rc"
  fi
  if [[ "${1:-}" == "__boot_inner" ]]; then
    internal_boot "$2" "$3" "$4"
    return $?
  fi
  [[ -x "$PY" ]] || { log "venv python missing: $PY"; return 2; }
  [[ -x "$BOX_GUARD" || -f "$BOX_GUARD" ]] || { log "box_guard missing: $BOX_GUARD"; return 2; }
  local -a steps=("$@")
  if (( ${#steps[@]} == 0 )) || [[ "${steps[0]}" == all ]]; then
    steps=("${ALL_STEPS[@]}")
  fi
  local step
  for step in "${steps[@]}"; do
    log "=== step $step"
    run_step "$step" || true
  done
}

main "$@"
