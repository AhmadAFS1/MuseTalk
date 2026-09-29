#!/usr/bin/env bash
# Standing video A/B validation: GPU sequence (pre-change baselines, then rounds r1_engines and r1_serving).
#
# Tool: scripts/video_ab.py (+ scripts/video_ab_chin_render.py, scripts/replay_scheduler_exactness.py).
# Every GPU step is its own lease: scripts/box_guard.sh run --min-avail-gb N --wait-min 60 --label video_ab_<step>
# (the lease honours /workspace/.gpu_lease.pause). Composition (A/B video, JSON, contact sheet) is CPU-only and
# runs outside the lease. Every step prints "STEP <name> PASS|FAIL|SKIP|BLOCKED rc=.." plus the tool's own
# PASS/FAIL lines with measured numbers, and writes JSON into this directory ($D).
#
# Usage:  gpu_sequence.sh [--print] [step ...]      (no step = all, in the order below)
#         --print only prints the commands (CPU, nothing runs).
# Override an arm's flags with <STEP>_FLAGS='K=V,...' (e.g. S3_FLAGS=...), the stagewise batch with STAGEWISE_BATCH.
#
# Arms: A = PRE-CHANGE = /workspace/MuseTalk at main HEAD, no new flags (cached per HEAD under
#       experiments/video_validation/baselines/<clip>/<head12>/). B = /workspace/MuseTalk-perf300 + flags.
# Clips: chin_japanese, chin_latina (TAESD + native encoder + 100% chin + refined seam, 240 frames @24 fps, 10 s)
#        replay_bob_mid (3-pose chinese_bob motion avatar, mid-turn pose switch), replay_jp_d10 (standard avatar)
#        (golden replay of all 10 harness jobs per arm; the two clips are recorded losslessly, <=20 s @20 fps)
# Outputs per clip: experiments/video_validation/<round>/<clip>__<arm>_ab.{mp4,json} + _contact.jpg; README index.
#
# Gates (see scripts/video_ab.py evaluate_gate):
#   exact : every compared pre-encoder frame SHA-256 identical, equal frame counts, and (replay) the harness golden
#           compare of ALL 10 jobs identical (faces, composed BGR, yuv420p, order, status). Diff panel black.
#   fp16  : [proposed, E1] full-frame min PSNR >= 40 dB, mouth-ROI min PSNR >= 36 dB, full mean |A-B| <= 0.5 LSB,
#           equal frame counts; chin clips also report FaceMesh jaw+lip deviation (G-TRACK proposal mean <= 0.05 px,
#           p99 <= 0.15 px, reported not gating). PASS still means "visual review required".
#   render: chin arm repeats must be SHA-identical (determinism) and recipe checks must pass (protected lip pixels
#           unchanged, ROI path == full-frame reference, Jacobian > .25); a requested backend that is not active
#           (silent fallback) is a FAIL.
#
# Steps (runtime estimates are for an idle RTX 4070 SUPER; torch.compile of TAESD adds ~1-3 min per chin process):
#   S0  CPU selftest (synthetic A/B + real dry-run on both trees) ............ ~20 s   -> selftest.json
#   B1  pre-change chin baselines (2 identities x 3 repeats), lease 9 GB ..... ~5 min  -> B1_baseline_chin.json
#       expect: PASS lines, 240 frames each, measured ~185-195 fps (Sep-27 run.py: 184.9 JP / 193.4 LAT),
#       repeats_identical=True, checks=True. No-op (SKIP-cached) when main's HEAD is unchanged.
#   B2  pre-change replay baselines (golden, 10 jobs), lease 11 GB .......... ~4 min  -> B2_baseline_replay.json
#       expect: PASS lines; aggregate golden fps printed (unpaced; ~200-260 fps).
#   E1  r1_engines/taesd_trt: MUSETALK_TAESD_BACKEND=trt (strict, no build) . ~9 min + 2 min compose, expect fp16
#       -> r1_engines__taesd_trt__render.json, r1_engines__taesd_trt.json. SKIP if models/taesd/trt is empty.
#   E2  r1_engines/unet_cudagraph: MUSETALK_TRT_UNET_CUDAGRAPHS=manual ...... ~9 min + 2 min compose, expect exact
#   E3  r1_engines/unet_stagewise: MUSETALK_UNET_BACKEND=trt_stagewise ...... ~9 min + 2 min compose, expect fp16
#       SKIP unless models/tensorrt_unet_stagewise_sm89/bs$STAGEWISE_BATCH has manifest.json + all 11 plans.
#   S1  r1_serving/wt_default: worktree, NO flags (default == today) ......... ~9 min + 2 min compose, expect exact
#   S2  r1_serving/sync_off: HLS_GPU_STAGE_SYNC_TIMING=0, MUSETALK_VAE_DECODE_TIMING_SYNC=0 (replay) ~4 min, exact
#   S3  r1_serving/mem_layout: avatar memory-layout flags (replay) .......... ~4 min + 1 min compose, expect exact
#   S4  r1_serving/yuv_compose: WEBRTC_YUV_IN_COMPOSE=1 (replay) ............ ~4 min + 1 min compose, expect exact
#   S5  r1_serving/serving_all: S2+S3+S4 flags together (replay) ............ ~4 min + 1 min compose, expect exact
#   X   rollup (CPU): one line per clip verdict + README index .............. ~5 s    -> rollup.json
# Total ~75 min of leases if nothing is cached. Disk: ~90 MB per lossless baseline clip; candidate lossless frames
# are deleted after a passing compose (their per-frame SHA-256 stay in clip.json; pass --keep-dumps to keep them).
set -uo pipefail
R=/workspace/MuseTalk-perf300
D=$R/docs/fps_comparisons/4070s_300fps_impl_20260928/video_ab
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
VAB="$PY $R/scripts/video_ab.py"
CHIN=chin_japanese,chin_latina
REPLAY=replay_bob_mid,replay_jp_d10
ALL=$CHIN,$REPLAY
STAGEWISE_BATCH=${STAGEWISE_BATCH:-16}
E1_FLAGS=${E1_FLAGS:-MUSETALK_TAESD_BACKEND=trt,MUSETALK_TAESD_TRT_STRICT=1,MUSETALK_TAESD_TRT_BUILD=0}
E2_FLAGS=${E2_FLAGS:-MUSETALK_TRT_UNET_CUDAGRAPHS=manual}
E3_FLAGS=${E3_FLAGS:-MUSETALK_UNET_BACKEND=trt_stagewise,MUSETALK_UNET_STAGEWISE_BATCH=$STAGEWISE_BATCH}
S2_FLAGS=${S2_FLAGS:-HLS_GPU_STAGE_SYNC_TIMING=0,MUSETALK_VAE_DECODE_TIMING_SYNC=0}
S3_FLAGS=${S3_FLAGS:-MUSETALK_AVATAR_MASK_CHANNELS=1,MUSETALK_AVATAR_FRAME_STORE=png,MUSETALK_AVATAR_MASK_STORE=png,MUSETALK_AVATAR_PLAN_FLOAT_ALPHA=0}
S4_FLAGS=${S4_FLAGS:-WEBRTC_YUV_IN_COMPOSE=1}
S5_FLAGS=${S5_FLAGS:-$S2_FLAGS,$S3_FLAGS,$S4_FLAGS}
PRINT=0
if [[ ${1:-} == --print ]]; then PRINT=1; shift; fi
mkdir -p "$D"
cd "$R" || exit 2

run() {  # run or print a command, logging to $D/<step>.log
    local step=$1; shift
    if (( PRINT )); then printf '[%s] %s\n' "$step" "$*"; return 0; fi
    echo "=== $step $(date -u +%H:%M:%S): $*" | tee -a "$D/$step.log"
    "$@" 2>&1 | tee -a "$D/$step.log"
    return "${PIPESTATUS[0]}"
}

lease() {  # lease <gb> <step> -- cmd...
    local gb=$1 step=$2; shift 3
    run "$step" "$R/scripts/box_guard.sh" run --min-avail-gb "$gb" --wait-min 60 --need-disk-gb 3 --label "video_ab_$step" -- "$@"
}

status() {  # status <step> <rc> [SKIP reason]
    local step=$1 rc=$2 word
    if [[ -n ${3:-} ]]; then word=SKIP
    elif [[ $rc == 0 ]]; then word=PASS
    elif [[ $rc == 75 || $rc == 76 ]]; then word=BLOCKED
    else word=FAIL; fi
    (( PRINT )) && return 0
    echo "STEP $step $word rc=$rc ${3:-}" | tee -a "$D/$step.log"
    if [[ $word == SKIP || $word == BLOCKED ]]; then
        printf '{"step": "%s", "result": "%s", "rc": %s, "reason": "%s", "utc": "%s"}\n' \
            "$step" "$word" "$rc" "${3:-lease/RAM wait timed out (box_guard rc $rc)}" "$(date -u +%FT%TZ)" > "$D/$step.json"
    fi
}

baseline() {  # baseline <step> <gb> <clips>
    local step=$1 gb=$2 clips=$3 rc
    if (( ! PRINT )) && $VAB baseline --clips "$clips" --check > "$D/$step.check.log" 2>&1; then
        cat "$D/$step.check.log"; status "$step" 0 "cached for main HEAD (no render needed)"; return 0
    fi
    lease "$gb" "$step" -- $VAB baseline --clips "$clips" --report "$D/$step.json"; rc=$?
    status "$step" $rc
}

arm() {  # arm <step> <round> <arm> <clips> <expect> <flags> [gb]
    local step=$1 round=$2 name=$3 clips=$4 expect=$5 flags=$6 gb=${7:-11} rc
    lease "$gb" "$step" -- $VAB render-arm --round "$round" --arm "$name" --clips "$clips" --flags "$flags" \
        --report "$D/${round}__${name}__render.json"; rc=$?
    if [[ $rc == 75 || $rc == 76 ]]; then status "$step" $rc; return; fi
    # composition is CPU-only: outside the lease; it also composes the clips that did render when others failed
    run "$step" $VAB compose-arm --round "$round" --arm "$name" --expect "$expect" --report "$D/${round}__${name}.json"
    local crc=$?
    (( rc == 0 )) && rc=$crc
    status "$step" $rc
}

stagewise_ready() {
    local dir=$R/models/tensorrt_unet_stagewise_sm89/bs$STAGEWISE_BATCH b
    [[ -f $dir/manifest.json ]] || return 1
    for b in head down0 down1 down2 down3 mid up0 up1 up2 up3 tail; do [[ -f $dir/$b.plan ]] || return 1; done
}

step() {
    case "$1" in
        S0) run S0 $VAB selftest --out "$D/selftest.json"; status S0 $? ;;
        B1) baseline B1_baseline_chin 9 "$CHIN" ;;
        B2) baseline B2_baseline_replay 11 "$REPLAY" ;;
        E1) if (( ! PRINT )) && [[ -z $(ls -A "$R/models/taesd/trt" 2>/dev/null) ]]; then
                status E1 0 "no TAESD TRT engine in models/taesd/trt (built by the taesd_trt area)"; return; fi
            arm E1 r1_engines taesd_trt "$ALL" fp16 "$E1_FLAGS" ;;
        E2) arm E2 r1_engines unet_cudagraph "$ALL" exact "$E2_FLAGS" ;;
        E3) if (( ! PRINT )) && ! stagewise_ready; then
                status E3 0 "stagewise engines bs$STAGEWISE_BATCH incomplete (unet_fp16/run_sequence.sh B2_$STAGEWISE_BATCH)"; return; fi
            arm E3 r1_engines unet_stagewise "$ALL" fp16 "$E3_FLAGS" ;;
        S1) arm S1 r1_serving wt_default "$ALL" exact "" ;;
        S2) arm S2 r1_serving sync_off "$REPLAY" exact "$S2_FLAGS" ;;
        S3) arm S3 r1_serving mem_layout "$REPLAY" exact "$S3_FLAGS" ;;
        S4) arm S4 r1_serving yuv_compose "$REPLAY" exact "$S4_FLAGS" ;;
        S5) arm S5 r1_serving serving_all "$REPLAY" exact "$S5_FLAGS" ;;
        X) run X $VAB rollup --dir "$D" --out "$D/rollup.json"; status X $? ;;
        *) echo "unknown step $1 (S0 B1 B2 E1 E2 E3 S1 S2 S3 S4 S5 X)"; exit 2 ;;
    esac
}

steps=("$@")
(( ${#steps[@]} )) || steps=(S0 B1 B2 E1 E2 E3 S1 S2 S3 S4 S5 X)
for s in "${steps[@]}"; do step "$s"; done
