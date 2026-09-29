#!/usr/bin/env bash
# GPU validation sequence for startup-rework component A (scripts/musetalk_host_profile.py: host facts,
# recipe resolution fast / fast300 / legacy_int8, verify-log). Written by the implementing agent, which
# was CPU-only: the GPU steps below have NOT been run yet.
#
# Usage:  gpu_sequence.sh [STEP ...]     (no args = every step, in order)
#         gpu_sequence.sh --list         the steps with runtime/RAM estimates
# Env:    PY=<venv python>               (default /workspace/.venvs/musetalk_trt_stagewise/bin/python)
#         WAIT_MIN=60                    box_guard lease wait per GPU step
#         LIVE_LOG=<server log> LIVE_OFFSET=<bytes> LIVE_RESOLVED=<resolved json>   for R5 (optional)
#
# The resolver itself never touches the GPU; what needs hardware is the claim it makes: "with this resolved
# env the server activates backend X". R2/R4 load exactly the backends a resolved env selects, through the
# same loader functions scripts/avatar_manager_parallel.py calls, print the same two log lines the server
# prints, and then run verify-log against them (backend_probe.py). Every GPU step is its own lease:
#   scripts/box_guard.sh run --min-avail-gb N --wait-min 60 --label startup_<step> -- <cmd>
# box_guard exit 75/76/77 (GPU busy / RAM / disk) is reported as SKIP, not FAIL. Every step prints one line
#   STEP <name> PASS|FAIL|SKIP rc=<rc> (<seconds>s) <note>
# JSON/log outputs land next to this script; the summary is gpu_sequence_summary.json.
#
# Nothing here writes <repo>/.runtime (on this box a symlink into /workspace/MuseTalk/.runtime): every
# resolve uses --out/--report into this directory, and the fast300 "all groups on" case uses a scratch
# copy of configs/recipes/fast300.env through MUSETALK_RECIPE_FILE (the tracked file is not edited).
#
# Steps (RTX 4070 SUPER estimates; RAM = peak host RAM of the step):
#   R0_cpu_tests           CPU  ~15 s  <0.3 GB  unit tests (66) + the launch chain's integration group (real resolver)
#   R1_resolve_fast        CPU  ~1 s   <0.1 GB  resolve --recipe fast for this box -> resolved_fast.{env,json}
#   R2_probe_fast          GPU  1-4 min 3 GB eager / ~10 GB with the .ts UNet (VmHWM 9.5 GB) - loads the fast
#                                       recipe's backends (compiled TAESD warm-up at the buckets = the slow part on a
#                                       cold inductor cache), one forward per bucket, then verify-log PASS/FAIL
#   R3_resolve_fast300_on  CPU  ~2 s   <0.1 GB  every @lever group of fast300.env switched on in a scratch copy: which
#                                       groups THIS box's engines/code allow, each drop with its reason (report only)
#   R4_probe_fast300_on    GPU  1-3 min ~3-4 GB  probe + verify-log for the R3 env when it enables an engine group
#                                       (stagewise UNet and/or TAESD TRT); SKIP when no engine group is enabled
#   R5_verify_live         CPU  <1 s   <0.1 GB  OPTIONAL: verify-log on a real server log (LIVE_LOG, LIVE_OFFSET)
set -Eeuo pipefail

REPO=${REPO:-/workspace/MuseTalk-perf300}
OUT=${OUT:-$REPO/docs/startup_rework_20260928/impl/resolver}
VENV=${VENV:-/workspace/.venvs/musetalk_trt_stagewise}
PY=${PY:-$VENV/bin/python}
GUARD=${GUARD:-$REPO/scripts/box_guard.sh}
WAIT_MIN=${WAIT_MIN:-60}
TOOL="$REPO/scripts/musetalk_host_profile.py"
PROBE=${PROBE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/backend_probe.py}
SUMMARY=$OUT/gpu_sequence_summary.json
declare -a RESULTS=()

log() { printf '[gpu_sequence %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
trap 'log "failed at line $LINENO: $BASH_COMMAND"' ERR

record() {  # record <step> <PASS|FAIL|SKIP> <rc> <seconds> <note>
    local step=$1 verdict=$2 rc=$3 secs=$4 note=${5:-}
    printf 'STEP %s %s rc=%s (%ss)%s\n' "$step" "$verdict" "$rc" "$secs" "${note:+ $note}"
    [[ "$rc" =~ ^[0-9]+$ ]] || rc=null
    RESULTS+=("$(printf '{"step":"%s","verdict":"%s","rc":%s,"seconds":%s,"note":"%s"}' \
        "$step" "$verdict" "$rc" "$secs" "${note//\"/\'}")")
}

guarded() {  # guarded <step> <min_avail_gb> -- cmd...  (GPU lease + RAM watchdog)
    local step=$1 min_avail=$2; shift 2
    [[ "$1" == "--" ]] && shift
    "$GUARD" run --min-avail-gb "$min_avail" --wait-min "$WAIT_MIN" --label "startup_$step" -- "$@"
}

json_get() {  # json_get <file> <python expression over d>
    "$PY" -I -B -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$1" "$2"
}

resolve_to() {  # resolve_to <name> <recipe> [ENV=VALUE ...] -> $OUT/resolved_<name>.{env,json}
    local name=$1 recipe=$2; shift 2
    env "$@" "$PY" -B "$TOOL" resolve --repo-root "$REPO" --venv "$VENV" --recipe "$recipe" \
        --out "$OUT/resolved_$name.env" --report "$OUT/resolved_$name.json" 2> "$OUT/resolved_$name.stderr"
}

probe_and_verify() {  # probe_and_verify <step> <name> <min_avail_gb>
    local step=$1 name=$2 min_avail=$3 rc=0 started secs
    started=$(date +%s)
    guarded "$step" "$min_avail" -- "$PY" -B "$PROBE" --repo-root "$REPO" --env "$OUT/resolved_$name.env" \
        --log "$OUT/probe_$name.log" --json "$OUT/probe_$name.json" > "$OUT/$step.log" 2>&1 || rc=$?
    secs=$(( $(date +%s) - started ))
    case "$rc" in
        75|76|77) record "$step" SKIP "$rc" "$secs" "box_guard refused (GPU busy / RAM / disk); see $step.log"; return 0 ;;
        0) ;;
        *) record "$step" FAIL "$rc" "$secs" "backend probe failed; see $step.log and probe_$name.json"; return 0 ;;
    esac
    rc=0
    "$PY" -B "$TOOL" verify-log --log "$OUT/probe_$name.log" --offset 0 --expect-vae auto --expect-unet auto \
        --resolved "$OUT/resolved_$name.json" --timeout 0 > "$OUT/verify_$name.json" 2>> "$OUT/$step.log" || rc=$?
    secs=$(( $(date +%s) - started ))
    if (( rc == 0 )); then
        record "$step" PASS 0 "$secs" "$(json_get "$OUT/verify_$name.json" '"vae=%s unet=%s" % (d["found"]["vae"], d["found"]["unet"])')"
    else
        record "$step" FAIL "$rc" "$secs" "verify-log rc=$rc; see verify_$name.json"
    fi
}

# ---------------------------------------------------------------------------------------------- steps
R0_cpu_tests() {
    local started rc=0
    started=$(date +%s)
    ( cd "$REPO" && python3 -B -m unittest test_musetalk_host_profile ) > "$OUT/R0_cpu_tests.log" 2>&1 || rc=$?
    if (( rc == 0 )); then
        ( cd "$REPO" && bash scripts/test_startup_scripts.sh -k integration ) >> "$OUT/R0_cpu_tests.log" 2>&1 || rc=$?
    fi
    record R0_cpu_tests "$( (( rc == 0 )) && echo PASS || echo FAIL)" "$rc" $(( $(date +%s) - started )) "see R0_cpu_tests.log"
}

R1_resolve_fast() {
    local started rc=0 note
    started=$(date +%s)
    resolve_to fast fast || rc=$?
    if (( rc == 0 )); then
        note="$(json_get "$OUT/resolved_fast.json" '"expect vae=%s unet=%s | %s" % (d["expect"]["vae"], d["expect"]["unet"], d["unet"].get("reason"))')"
        record R1_resolve_fast PASS 0 $(( $(date +%s) - started )) "$note"
    else
        record R1_resolve_fast FAIL "$rc" $(( $(date +%s) - started )) "see resolved_fast.stderr"
    fi
}

R2_probe_fast() {
    [[ -f "$OUT/resolved_fast.json" ]] || R1_resolve_fast
    local unet need=6
    unet="$(json_get "$OUT/resolved_fast.json" 'd["expect"]["unet"]')"
    [[ "$unet" == trt ]] && need=12  # the 2.2 GB .ts load peaks at ~9.5 GB host RSS
    probe_and_verify R2_probe_fast fast "$need"
}

R3_resolve_fast300_on() {
    local started rc=0 scratch="$OUT/fast300_all_groups_on.env" note
    started=$(date +%s)
    # Switch on every '#KEY=VALUE' line inside an '# @lever' block (the orchestrator's action), in a copy.
    awk '/^#[[:space:]]*@lever[[:space:]]/ {g=1; print; next}
         /^[[:space:]]*$/ {g=0}
         g && /^#[A-Z][A-Z0-9_]*=/ {sub(/^#/, "")}
         {print}' "$REPO/configs/recipes/fast300.env" > "$scratch"
    resolve_to fast300_on fast300 "MUSETALK_RECIPE_FILE=$scratch" || rc=$?
    if (( rc == 0 )); then
        note="$(json_get "$OUT/resolved_fast300_on.json" '"enabled=%s dropped=%s" % ([g["name"] for g in d["recipe_groups"] if g["status"]=="enabled"], [g["name"] for g in d["recipe_groups"] if g["status"]=="dropped"])')"
        record R3_resolve_fast300_on PASS 0 $(( $(date +%s) - started )) "$note"
    else
        record R3_resolve_fast300_on FAIL "$rc" $(( $(date +%s) - started )) "see resolved_fast300_on.stderr"
    fi
}

R4_probe_fast300_on() {
    [[ -f "$OUT/resolved_fast300_on.json" ]] || R3_resolve_fast300_on
    local engines need=6
    engines="$(json_get "$OUT/resolved_fast300_on.json" '",".join(g["name"] for g in d["recipe_groups"] if g["status"]=="enabled" and g["name"] in ("stagewise_unet","taesd_trt","ts_unet_cudagraphs"))')"
    if [[ -z "$engines" ]]; then
        record R4_probe_fast300_on SKIP - 0 "no engine lever group is enabled on this box (see R3 reasons)"
        return 0
    fi
    [[ "$(json_get "$OUT/resolved_fast300_on.json" 'd["expect"]["unet"]')" == trt ]] && need=12
    probe_and_verify R4_probe_fast300_on fast300_on "$need"
}

R5_verify_live() {
    if [[ -z "${LIVE_LOG:-}" ]]; then
        record R5_verify_live SKIP - 0 "set LIVE_LOG (+ LIVE_OFFSET, LIVE_RESOLVED) to check a real server log"
        return 0
    fi
    local rc=0 started
    started=$(date +%s)
    "$PY" -B "$TOOL" verify-log --log "$LIVE_LOG" --offset "${LIVE_OFFSET:-0}" --expect-vae auto --expect-unet auto \
        --resolved "${LIVE_RESOLVED:-$REPO/.runtime/musetalk_resolved.json}" --timeout 0 > "$OUT/verify_live.json" || rc=$?
    record R5_verify_live "$( (( rc == 0 )) && echo PASS || echo FAIL)" "$rc" $(( $(date +%s) - started )) "see verify_live.json"
}

ALL_STEPS=(R0_cpu_tests R1_resolve_fast R2_probe_fast R3_resolve_fast300_on R4_probe_fast300_on R5_verify_live)

if [[ "${1:-}" == "--list" ]]; then
    sed -n '/^# Steps/,/^set -Eeuo/p' "$0" | sed '$d'
    exit 0
fi
[[ -x "$PY" ]] || die "venv python not found: $PY"
[[ -f "$TOOL" && -f "$PROBE" ]] || die "missing $TOOL or $PROBE"
[[ -x "$GUARD" ]] || die "box_guard not found: $GUARD"
mkdir -p "$OUT"
steps=("$@")
(( ${#steps[@]} )) || steps=("${ALL_STEPS[@]}")
for step in "${steps[@]}"; do
    declare -F "$step" >/dev/null || die "unknown step $step (see --list)"
    log "=== $step"
    "$step"
done
{
    printf '{"schema":"musetalk_startup_gpu_sequence_v1","component":"resolver","finished_utc":"%s","steps":[' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ( IFS=,; printf '%s' "${RESULTS[*]}" )
    printf ']}\n'
} > "$SUMMARY"
log "summary: $SUMMARY"
if printf '%s\n' "${RESULTS[@]}" | grep -q '"verdict":"FAIL"'; then
    exit 1
fi
exit 0
