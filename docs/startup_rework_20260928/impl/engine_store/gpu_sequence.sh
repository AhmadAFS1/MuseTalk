#!/usr/bin/env bash
# GPU validation sequence for component B (engine store: scripts/unet_engine_store.py +
# scripts/musetalk_engine_keys.py + calibration/unet_portable_bs8). Written by the implementing
# agent, which was CPU-only: nothing below has been run on the GPU yet.
#
# Usage:  gpu_sequence.sh [STEP ...]        (no args = the default steps, in order)
#         gpu_sequence.sh --list            print the steps with runtime/RAM estimates
# Env:    PY=<venv python>                  (default /workspace/.venvs/musetalk_trt_stagewise/bin/python)
#         RUN_TS_BUILD=1                    also run the optional 7-8 min .ts build (scratch store)
#         RUN_STAGEWISE_BUILD=1             also run the optional ~9 min stagewise bs8 build (scratch store)
#         WAIT_MIN=60                       box_guard lease wait per step
#
# Every GPU step is its own box_guard lease:
#   scripts/box_guard.sh run --min-avail-gb N --wait-min 60 --label startup_<step> -- <cmd>
# and prints one line "STEP <name> PASS|FAIL|SKIP rc=<rc> (<seconds>s)"; box_guard exit 75/76/77
# (GPU busy / RAM / disk) is reported as SKIP, not FAIL. JSON results land next to this script as
# <step>.json, logs as <step>.log, and a summary in gpu_sequence_summary.json.
#
# Where engines land: the DEFAULT stores (what the resolver reads) under <repo>/models/. On this box
# <repo>/models is a symlink to /workspace/MuseTalk/models (gitignored, shared with main). Adopt only
# ADDS new directories holding symlinks + small JSON files there (the original engines are never
# modified); the TAESD build adds ~6 MB. Optional builds use a scratch store under /workspace/tmp and
# delete it afterwards.
#
# Store validation = integrity/accuracy of the engine ON THIS GPU. It is not the fast300 quality gate for
# TAESD TRT: `list`/`find_engine` report entry["quality_gate"]["verdict"] from the engine meta (G-TAESD,
# written by docs/fps_comparisons/4070s_300fps_impl_20260928/taesd_trt/gate_taesd_trt.py). The flat engine
# taesd_trt_6111388248264a4ef2ae built on 2026-09-28 06:04 records verdict FAIL (full-face max 5 LSB > 3),
# so S6 makes it store-usable while MUSETALK_TAESD_BACKEND=trt must stay off until a PASS is recorded.
#
# Steps (estimates for the RTX 4070 SUPER box; RAM = peak host RAM of the step):
#   S0_cpu_checks       CPU  ~15 s  <0.5 GB  unit tests, py_compile, check-corpus, list --scan (scans 2 .ts, ~7 s)
#   S1_adopt_ts         CPU  ~5 s   <0.3 GB  adopt models/tensorrt_unet_sm89_bs8_local/unet_trt.ts --no-validate
#   S2_validate_ts      GPU  ~2 min ~9.5 GB  validate unet_ts (.ts load drops MemAvailable ~8.4 GB; lease waits for 13 GB) -> recipe fast
#   S3_validate_ts_cg   GPU  ~2 min ~9.5 GB  validate unet_ts --cudagraphs manual (fast300 lever MUSETALK_TRT_UNET_CUDAGRAPHS)
#   S4_adopt_stagewise  CPU  ~2 s   <0.3 GB  adopt models/tensorrt_unet_stagewise_sm89/bs16 --no-validate
#   S5_validate_sw16    GPU  ~1 min ~2 GB    validate unet_stagewise bs16 (probe hash + portable corpus; 1.29 GB measured on full corpus)
#   S6_taesd_trt_bs8    GPU  ~2 min ~3 GB    ensure taesd_trt bs8 (adopts flat models/taesd/trt engines if present, else builds ~15-60 s)
#   S7_ensure_all       CPU  ~3 s   <0.3 GB  ensure --provision off for all three kinds: expect rc 0 each (what the resolver will find)
#   S8_remote_taesd     GPU  ~1 min ~3 GB    file:// publish of the TAESD entry + restore into a scratch store + validate
#   S9_build_ts         GPU  ~8 min ~11 GB   OPTIONAL (RUN_TS_BUILD=1): build unet_ts from scratch into a scratch store, +2.2 GB disk
#   S10_build_sw8       GPU  ~10 min ~9.5 GB OPTIONAL (RUN_STAGEWISE_BUILD=1): build unet_stagewise bs8 into a scratch store, +0.9 GB disk
set -Eeuo pipefail

# REPO/OUT/VENV/GUARD are overridable only for a CPU dry run of this script's plumbing.
REPO=${REPO:-/workspace/MuseTalk-perf300}
OUT=${OUT:-$REPO/docs/startup_rework_20260928/impl/engine_store}
VENV=${VENV:-/workspace/.venvs/musetalk_trt_stagewise}
PY=${PY:-$VENV/bin/python}
GUARD=${GUARD:-$REPO/scripts/box_guard.sh}
WAIT_MIN=${WAIT_MIN:-60}
SCRATCH=${SCRATCH:-/workspace/tmp/startup_engine_store_$$}
STORE_CLI=("$PY" -B "$REPO/scripts/unet_engine_store.py")
COMMON=(--repo-root "$REPO" --venv "$VENV")
# flat strings for the multi-command `bash -c` steps (these paths contain no spaces)
SC="${STORE_CLI[*]}"
CM="${COMMON[*]}"
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

# run_step <step> <expected rc list e.g. "0" or "0 3"> <gpu:0|1> <min_avail_gb> <need_disk_gb> -- cmd...
run_step() {
    local step=$1 expect=$2 gpu=$3 min_avail=$4 need_disk=$5; shift 5
    [[ "$1" == "--" ]] && shift
    local started rc=0 verdict note=""
    started=$(date +%s)
    log "=== $step: $*"
    if [[ "$gpu" == 1 ]]; then
        "$GUARD" run --min-avail-gb "$min_avail" --need-disk-gb "$need_disk" --wait-min "$WAIT_MIN" \
            --label "startup_$step" -- "$@" >"$OUT/$step.stdout" 2>"$OUT/$step.log" || rc=$?
    else
        "$@" >"$OUT/$step.stdout" 2>"$OUT/$step.log" || rc=$?
    fi
    verdict=FAIL
    for want in $expect; do [[ "$rc" == "$want" ]] && verdict=PASS; done
    if [[ "$gpu" == 1 && "$verdict" == FAIL ]]; then
        case $rc in
            75) verdict=SKIP; note="box_guard: GPU not quiet within ${WAIT_MIN} min (retry later)" ;;
            76) verdict=SKIP; note="box_guard: MemAvailable < ${min_avail} GB" ;;
            77) verdict=SKIP; note="box_guard: disk free < ${need_disk} GB" ;;
            86|87) note="box_guard: RAM watchdog/oom_kill (rc $rc)" ;;
        esac
    fi
    [[ -s "$OUT/$step.stdout" ]] && cp "$OUT/$step.stdout" "$OUT/$step.json" 2>/dev/null || true
    rm -f "$OUT/$step.stdout"
    record "$step" "$verdict" "$rc" "$(( $(date +%s) - started ))" "$note"
    [[ "$verdict" != FAIL ]]
}

S0_cpu_checks() {
    run_step S0_cpu_checks 0 0 0 0 -- bash -c "
        set -Eeuo pipefail
        cd $REPO
        $PY -B -m py_compile scripts/musetalk_engine_keys.py scripts/unet_engine_store.py test_unet_engine_store.py
        $PY -B -m unittest test_unet_engine_store 1>&2
        $SC check-corpus $CM >/dev/null
        $SC list --scan $CM"
}
S1_adopt_ts() {  # rc 3 = registered, not yet validated; 0 = already adopted and validated earlier
    run_step S1_adopt_ts "0 3" 0 0 0 -- "${STORE_CLI[@]}" adopt --kind unet_ts \
        --ts "$REPO/models/tensorrt_unet_sm89_bs8_local/unet_trt.ts" --no-validate "${COMMON[@]}"
}
S2_validate_ts() {
    run_step S2_validate_ts 0 1 13 1 -- "${STORE_CLI[@]}" validate --kind unet_ts "${COMMON[@]}"
}
S3_validate_ts_cg() {
    run_step S3_validate_ts_cg 0 1 13 1 -- "${STORE_CLI[@]}" validate --kind unet_ts --cudagraphs manual "${COMMON[@]}"
}
S4_adopt_stagewise() {
    run_step S4_adopt_stagewise "0 3" 0 0 0 -- "${STORE_CLI[@]}" adopt --kind unet_stagewise --batch 16 \
        --dir "$REPO/models/tensorrt_unet_stagewise_sm89/bs16" --no-validate "${COMMON[@]}"
}
S5_validate_sw16() {
    run_step S5_validate_sw16 0 1 6 1 -- "${STORE_CLI[@]}" validate --kind unet_stagewise --batch 16 "${COMMON[@]}"
}
S6_taesd_trt_bs8() {  # no remote for this step: adopt flat engines if the other session built them, else build
    run_step S6_taesd_trt_bs8 0 1 6 2 -- env -u TRT_ARTIFACT_S3_BUCKET -u MUSETALK_ENGINE_REMOTE_BASE \
        -u MUSETALK_TAESD_TRT_ENGINE_REMOTE "${STORE_CLI[@]}" ensure --kind taesd_trt --batch 8 --provision auto "${COMMON[@]}"
}
S7_ensure_all() {  # rc 0 for every kind = the resolver will find a validated engine of each kind
    run_step S7_ensure_all 0 0 0 0 -- bash -c "
        set -Eeuo pipefail
        for spec in 'unet_ts 8' 'unet_stagewise 16' 'taesd_trt 8'; do
            set -- \$spec
            $SC ensure --kind \$1 --batch \$2 --provision off $CM > $OUT/S7_ensure_\$1.json
        done
        $SC list $CM"
}
S8_remote_taesd() {
    local rc=0
    mkdir -p "$SCRATCH"
    run_step S8_remote_taesd 0 1 6 2 -- bash -c "
        set -Eeuo pipefail
        $SC publish --kind taesd_trt --batch 8 --remote file://$SCRATCH/remote $CM >/dev/null
        $SC restore --kind taesd_trt --batch 8 --remote file://$SCRATCH/remote --store $SCRATCH/store_taesd $CM" || rc=$?
    rm -rf "$SCRATCH/remote" "$SCRATCH/store_taesd"
    return "$rc"
}
S9_build_ts() {
    if [[ "${RUN_TS_BUILD:-0}" != 1 ]]; then record S9_build_ts SKIP - 0 "set RUN_TS_BUILD=1 to run"; return 0; fi
    local rc=0
    mkdir -p "$SCRATCH"
    run_step S9_build_ts 0 1 14 6 -- "${STORE_CLI[@]}" build --kind unet_ts --store "$SCRATCH/store_ts" \
        --timeout-min 30 "${COMMON[@]}" || rc=$?
    rm -rf "$SCRATCH/store_ts"
    return "$rc"
}
S10_build_sw8() {
    if [[ "${RUN_STAGEWISE_BUILD:-0}" != 1 ]]; then record S10_build_sw8 SKIP - 0 "set RUN_STAGEWISE_BUILD=1 to run"; return 0; fi
    local rc=0
    mkdir -p "$SCRATCH"
    run_step S10_build_sw8 0 1 12 3 -- "${STORE_CLI[@]}" build --kind unet_stagewise --batch 8 \
        --store "$SCRATCH/store_sw" --timeout-min 45 "${COMMON[@]}" || rc=$?
    rm -rf "$SCRATCH/store_sw"
    return "$rc"
}

DEFAULT_STEPS=(S0_cpu_checks S1_adopt_ts S2_validate_ts S3_validate_ts_cg S4_adopt_stagewise S5_validate_sw16
               S6_taesd_trt_bs8 S7_ensure_all S8_remote_taesd S9_build_ts S10_build_sw8)

main() {
    if [[ "${1:-}" == "--list" ]]; then grep -E '^#   S[0-9]' "$0" | sed 's/^#   //'; return 0; fi
    [[ -x "$PY" ]] || die "venv python not found: $PY"
    [[ -x "$GUARD" ]] || die "box_guard not found: $GUARD"
    local free_gb
    free_gb=$(df -Pk "$REPO" | awk 'NR==2 {printf "%d", $4/1048576}')
    (( free_gb >= 12 )) || die "only ${free_gb} GB free on /workspace (keep >= 12 GB)"
    mkdir -p "$OUT"
    local steps=("$@")
    [[ ${#steps[@]} -gt 0 ]] || steps=("${DEFAULT_STEPS[@]}")
    local failed=0 step
    for step in "${steps[@]}"; do
        declare -F "$step" >/dev/null || die "unknown step $step (see --list)"
        "$step" || failed=$(( failed + 1 ))
    done
    rmdir "$SCRATCH" 2>/dev/null || true
    { printf '{"sequence":"engine_store","finished_utc":"%s","failed":%s,"steps":[' "$(date -u +%FT%TZ)" "$failed"
      local IFS=,; printf '%s' "${RESULTS[*]}"; printf ']}\n'; } > "$SUMMARY"
    log "summary: $SUMMARY (failed=$failed)"
    if (( failed == 0 )); then echo "GPU_SEQUENCE engine_store PASS"; else echo "GPU_SEQUENCE engine_store FAIL ($failed step(s))"; fi
    (( failed == 0 ))
}

main "$@"
