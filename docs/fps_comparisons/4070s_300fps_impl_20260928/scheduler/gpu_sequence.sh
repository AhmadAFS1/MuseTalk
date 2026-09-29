#!/usr/bin/env bash
# Scheduler pipeline GPU sequence (300 fps plan items 0.2, 1.1, 1.2 hookup, 1.4, 1.9, 1.10,
# crossfade-copy skip, WEBRTC_YUV_IN_COMPOSE producer side). Written 2026-09-28 by the
# CPU-only scheduler workflow; NOTHING here has been run on the GPU yet.
#
# Usage:  gpu_sequence.sh [step ...]      (no step = all, in order)
#         FORCE=1 gpu_sequence.sh g3      re-run a GPU step even if its JSON exists
# Every GPU step is its own box_guard lease (box_guard.sh run --min-avail-gb N --wait-min 60
# --label scheduler_<step> --kill-below-gb 4; the lease also honours /workspace/.gpu_lease.pause).
# Steps whose result JSON already exists (with "finished_at") are not re-run, only re-gated.
# Every step prints PASS/FAIL lines with measured numbers; gate JSON lands in this directory
# (gate_<name>.json), harness JSON as golden_*.json / speed_*.json, logs as *.log.
#
# Code under test: --repo /workspace/MuseTalk (clean main = today's behaviour, never modified)
# vs --repo /workspace/MuseTalk-perf300 (this worktree). Golden = 10 jobs submitted at once
# (3-pose motion "bob" x4 incl. a mid-turn pose change and an exact-silence turn, standard
# "jp" x3 incl. exact silence, "latfh1" x2), ~1550 frames; every decoded face, composed
# pre-encoder BGR frame and its PyAV yuv420p frame is SHA-256'd in order.
#
# step        lease  RAM   est. time  what / PASS threshold
# ----------  -----  ----  ---------  ---------------------------------------------------------------
# cpu         no     <2GB  ~1 min     harness selfcheck on both trees (imports come from --repo, fixtures,
#                                     WAV prep, no CUDA) + 27 CPU unit tests. PASS = all ok.
# g1          yes    15GB  ~6 min     MAIN: main_r1, main_r1b (same process), main_rawidle
#                                     (WEBRTC_RAW_IDLE_POSE=1); mp4 crf12 clips. PASS = main_r1b identical.
# g2          yes    15GB  ~4 min     MAIN fresh process: main_r2. PASS = identical to main_r1
#                                     ("golden baseline x2 on main must reproduce"). If g1/g2 FAIL, stop:
#                                     no candidate gate below is meaningful.
# g3          yes    15GB  ~11 min    WORKTREE, each flag alone: default, HLS_GPU_EVENT_TIMING=1,
#                                     HLS_GPU_PIPELINE_DEPTH=2, HLS_SCHEDULER_POLICY=edf,
#                                     HLS_SKIP_GPU_FOR_RAW=1, HLS_SKIP_CROSSFADE_COPY=1,
#                                     WEBRTC_YUV_IN_COMPOSE=1, MUSETALK_WHISPER_STREAM=1.
#                                     PASS = every run SHA-identical to main_r1 (frames + yuv + faces;
#                                     skipped raw faces excluded), 0 yuv-contract mismatches.
#                                     Also a non-gating diagnostic arm (today's scheduler inside the
#                                     worktree) that attributes a wt_default diff to scheduler vs other
#                                     WIP modules.
# g4          yes    15GB  ~8 min     WORKTREE: rawidle (default path, WEBRTC_RAW_IDLE_POSE=1),
#                                     rawidle+skip, all flags, all+rawidle, EDF with the wall-clock slack
#                                     estimate and a 1 s run-ahead cap, and a single-stream event-timing
#                                     run. PASS = identical to main_r1 / main_rawidle; evsync: event UNet
#                                     and VAE(+D2H) totals within 2% of the synced host totals (item 0.2).
# g5          yes    15GB  ~5 min     WORKTREE process env HLS_GPU_STAGE_SYNC_TIMING=0
#                                     MUSETALK_VAE_DECODE_TIMING_SYNC=0 (item 1.1, import-time flag):
#                                     nosync, all flags (+ clips). PASS = identical to main_r1.
# g6          yes    15GB  ~5 min     WORKTREE process env MUSETALK_TRT_UNET_CUDAGRAPHS=manual (item 1.2,
#                                     read at model load): depth 1 and depth 2. PASS = identical to
#                                     main_r1. A FAIL here indicts the UNet graph (1.2), not the scheduler.
# t1          yes    16GB  ~7 min     MAIN unpaced null-sink throughput, N=8/12/16, 15 s warmup + 60 s
#                                     window each (today's env). Baseline numbers only (PASS = ran).
# t2          yes    16GB  ~11 min    WORKTREE (stage syncs off) all flags, and depth2+events alone, same
#                                     N/window. PASS = candidate fps >= 1.00x main at every N with
#                                     >= 60 s windows; GOAL (reported, not gating) >= 1.10x. Capacity
#                                     telemetry (GPU busy fraction, idle gap, callback ms, feeder CPU)
#                                     is in speed_wt.json.
# video       no     <1GB  ~1 min     labelled side-by-side main_r1 | wt_all | 8x|diff| for bob_t1,
#                                     bob_mid, jp_d10 (tmp/scheduler_videos/, git-ignored). PASS = built.
# summary     no     -     secs       gpu_sequence_summary.json over every gate_*.json.
#
# Total: ~60 min of leases (8 separate leases, each < 15 min) + ~2 min CPU.
set -uo pipefail
R=/workspace/MuseTalk-perf300
MAIN=/workspace/MuseTalk
D=$R/docs/fps_comparisons/4070s_300fps_impl_20260928/scheduler
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
H=$R/scripts/replay_scheduler_exactness.py
GATE="$PY $D/gate.py"
VID=${SCHED_VIDEO_DIR:-$R/tmp/scheduler_videos}
CLIPS=bob_t1,bob_mid,jp_d10
export REPLAY_SCRATCH=${REPLAY_SCRATCH:-/tmp/claude-0/replay_scheduler}
cd "$R" || exit 2
mkdir -p "$VID" "$REPLAY_SCRATCH"

# Every in-run lever at once (HLS_GPU_STAGE_SYNC_TIMING is read in the scheduler's __init__).
ALL="HLS_GPU_EVENT_TIMING=1,HLS_GPU_STAGE_SYNC_TIMING=0,HLS_GPU_PIPELINE_DEPTH=2,HLS_SCHEDULER_POLICY=edf"
ALL="$ALL,HLS_SKIP_GPU_FOR_RAW=1,HLS_SKIP_CROSSFADE_COPY=1,WEBRTC_YUV_IN_COMPOSE=1,MUSETALK_WHISPER_STREAM=1"

done_json() { [[ -z ${FORCE:-} && -s $1 ]] && grep -q '"finished_at"' "$1"; }

harness() {  # harness <step> <gb> <out.json> [env K=V ...] -- harness args...
  local step=$1 gb=$2 out=$3; shift 3
  local envs=()
  while [[ $# -gt 0 && $1 != -- ]]; do envs+=("$1"); shift; done
  shift
  if done_json "$out"; then echo "=== $step: $out exists, re-gating only (FORCE=1 to re-run)"; return 0; fi
  local log=${out%.json}.log rc
  echo "=== $step: running under the lease -> $out (log $log)"
  env ${envs[@]+"${envs[@]}"} "$R/scripts/box_guard.sh" run --min-avail-gb "$gb" --wait-min 60 --kill-below-gb 4 \
    --label "scheduler_$step" -- $PY "$H" "$@" --out "$out" > "$log" 2>&1
  rc=$?
  printf '{"step": "%s", "rc": %d, "out": "%s", "log": "%s", "at": "%s"}\n' \
    "$step" "$rc" "$out" "$log" "$(date -u +%FT%TZ)" > "$D/step_${step}_$(basename "${out%.json}").rc.json"
  if [[ $rc -ne 0 ]]; then
    echo "FAIL $step: harness/box_guard rc=$rc (75 lease wait timeout, 76 RAM, 86 watchdog kill, 87 oom_kill; see $log)"
    grep -E "^\[replay\]|Error|Traceback" "$log" | tail -5
  fi
  return $rc
}

step_cpu() {
  local rc=0 out avail
  avail=$(awk '/^MemAvailable:/ {printf "%d", $2/1048576}' /proc/meminfo)
  if [[ $avail -lt 5 ]]; then echo "FAIL cpu: MemAvailable ${avail} GB < 5 GB; not starting CPU tests"; return 1; fi
  for repo in "$MAIN" "$R"; do
    out=$D/selfcheck_$(basename "$repo").json
    CUDA_VISIBLE_DEVICES="" $PY "$H" --repo "$repo" --mode selfcheck --out "$out" 2>&1 | grep -E "^(PASS|FAIL)" || rc=1
  done
  CUDA_VISIBLE_DEVICES="" $PY -m unittest scripts.test_hls_scheduler_pipeline > "$D/cpu_tests.log" 2>&1
  local trc=$?
  local summary; summary=$(grep -E "^(Ran|OK|FAILED)" "$D/cpu_tests.log" | tr '\n' ' ')
  printf '{"step": "cpu", "unit_tests_rc": %d, "summary": "%s", "at": "%s"}\n' "$trc" "$summary" \
    "$(date -u +%FT%TZ)" > "$D/gate_cpu_unit_tests.json.tmp"
  $PY - "$D/gate_cpu_unit_tests.json.tmp" "$trc" <<'EOF'
import json, sys
p, rc = sys.argv[1], int(sys.argv[2])
d = json.loads(open(p).read()); d.update({"gate": "cpu_unit_tests", "verdict": "PASS" if rc == 0 else "FAIL"})
open(p[:-4], "w").write(json.dumps(d, indent=1))
EOF
  rm -f "$D/gate_cpu_unit_tests.json.tmp"
  if [[ $trc -eq 0 ]]; then echo "PASS cpu_unit_tests: $summary"; else echo "FAIL cpu_unit_tests: $summary (see cpu_tests.log)"; rc=1; fi
  return $rc
}

step_g1() {
  harness g1 15 "$D/golden_main_r1.json" -- --repo "$MAIN" --mode golden \
    --run main_r1:repo --run main_r1b:repo --run main_rawidle:repo:WEBRTC_RAW_IDLE_POSE=1 \
    --video-dir "$VID" --video-jobs "$CLIPS" --video-seconds 12 --video-crf 12
  $GATE reproduce main_in_process "$D/golden_main_r1.json:main_r1" "$D/golden_main_r1.json:main_r1b"
}

step_g2() {
  harness g2 15 "$D/golden_main_r2.json" -- --repo "$MAIN" --mode golden --run main_r2:repo
  $GATE reproduce main_fresh_process "$D/golden_main_r1.json:main_r1" "$D/golden_main_r2.json:main_r2"
}

step_g3() {
  harness g3 15 "$D/golden_wt_alone.json" -- --repo "$R" --mode golden \
    --run wt_default:repo \
    --run wt_basesched:base \
    --run wt_event:repo:HLS_GPU_EVENT_TIMING=1 \
    --run wt_depth2:repo:HLS_GPU_PIPELINE_DEPTH=2 \
    --run wt_edf:repo:HLS_SCHEDULER_POLICY=edf \
    --run wt_skipraw:repo:HLS_SKIP_GPU_FOR_RAW=1 \
    --run wt_xfade:repo:HLS_SKIP_CROSSFADE_COPY=1 \
    --run wt_yuv:repo:WEBRTC_YUV_IN_COMPOSE=1 \
    --run wt_whisper:repo:MUSETALK_WHISPER_STREAM=1
  $GATE exact wt_default_vs_main "$D/golden_main_r1.json:main_r1" "$D/golden_wt_alone.json:wt_default"
  # Diagnostic for a wt_default FAIL (not gating): today's scheduler inside the worktree's other
  # modules. Identical to main -> the other WIP modules are equivalent and any diff is the scheduler;
  # different -> the diff comes from a non-scheduler module (api_avatar/vae/trt_runtime/...).
  $GATE exact diag_worktree_modules_with_base_scheduler "$D/golden_main_r1.json:main_r1" \
    "$D/golden_wt_alone.json:wt_basesched" || true
  $GATE exact wt_each_flag_alone_vs_main "$D/golden_main_r1.json:main_r1" \
    "$D/golden_wt_alone.json:wt_event+wt_depth2+wt_edf+wt_skipraw+wt_xfade+wt_yuv+wt_whisper"
}

step_g4() {
  harness g4 15 "$D/golden_wt_combo.json" -- --repo "$R" --mode golden \
    --run wt_rawidle:repo:WEBRTC_RAW_IDLE_POSE=1 \
    --run wt_rawidle_skip:repo:WEBRTC_RAW_IDLE_POSE=1,HLS_SKIP_GPU_FOR_RAW=1 \
    --run "wt_all_inrun:repo:$ALL" \
    --run "wt_all_rawidle:repo:$ALL,WEBRTC_RAW_IDLE_POSE=1" \
    --run wt_edf_wallclock:repo:HLS_SCHEDULER_POLICY=edf,HLS_SCHEDULER_MAX_RUNAHEAD_S=1,HLS_GPU_EVENT_TIMING=1,REPLAY_CONSUMER_DEPTH=none \
    --run wt_evsync:repo:HLS_GPU_EVENT_TIMING=1,REPLAY_JOBS=bob_t1
  $GATE exact wt_rawidle_vs_main_rawidle "$D/golden_main_r1.json:main_rawidle" \
    "$D/golden_wt_combo.json:wt_rawidle+wt_rawidle_skip+wt_all_rawidle"
  $GATE exact wt_all_inrun_vs_main "$D/golden_main_r1.json:main_r1" \
    "$D/golden_wt_combo.json:wt_all_inrun+wt_edf_wallclock+wt_evsync"
  $GATE evsync event_vs_sync_single_stream "$D/golden_wt_combo.json:wt_evsync" --tol 0.02
}

step_g5() {
  harness g5 15 "$D/golden_wt_nosync.json" HLS_GPU_STAGE_SYNC_TIMING=0 MUSETALK_VAE_DECODE_TIMING_SYNC=0 -- \
    --repo "$R" --mode golden --run wt_nosync:repo --run "wt_all:repo:$ALL" \
    --video-dir "$VID" --video-jobs "$CLIPS" --video-seconds 12 --video-crf 12
  $GATE exact wt_nosync_and_all_vs_main "$D/golden_main_r1.json:main_r1" "$D/golden_wt_nosync.json:wt_nosync+wt_all"
}

step_g6() {
  harness g6 15 "$D/golden_wt_cudagraph.json" MUSETALK_TRT_UNET_CUDAGRAPHS=manual -- \
    --repo "$R" --mode golden --run wt_graph_d1:repo \
    --run wt_graph_d2:repo:HLS_GPU_PIPELINE_DEPTH=2,HLS_GPU_EVENT_TIMING=1,HLS_GPU_STAGE_SYNC_TIMING=0
  $GATE exact wt_unet_cudagraph_manual_vs_main "$D/golden_main_r1.json:main_r1" \
    "$D/golden_wt_cudagraph.json:wt_graph_d1+wt_graph_d2"
}

step_t1() {
  harness t1 16 "$D/speed_main.json" -- --repo "$MAIN" --mode speed --run main_speed:repo \
    --n-jobs 8,12,16 --seconds 60 --warmup-s 15
  local rc=$?
  $PY - "$D/speed_main.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for r in d["runs"]:
    rows = " ".join(f"N={s['n_jobs']}:{s['generated_fps']}fps(util {s['smi'].get('util_pct')}%)" for s in r.get("speed", []))
    print(("PASS" if r.get("speed") and not r.get("error") else "FAIL") + f" speed_baseline {r['label']}: {rows} {r.get('error') or ''}")
EOF
  return $rc
}

step_t2() {
  harness t2 16 "$D/speed_wt.json" HLS_GPU_STAGE_SYNC_TIMING=0 MUSETALK_VAE_DECODE_TIMING_SYNC=0 -- \
    --repo "$R" --mode speed --run "wt_all_speed:repo:$ALL" \
    --run wt_depth2_speed:repo:HLS_GPU_PIPELINE_DEPTH=2,HLS_GPU_EVENT_TIMING=1 \
    --n-jobs 8,12,16 --seconds 60 --warmup-s 15
  $GATE speed throughput_null_sink "$D/speed_main.json:main_speed" "$D/speed_wt.json:wt_all_speed+wt_depth2_speed" \
    --min-ratio 1.00 --goal-ratio 1.10 --min-window-s 60
}

step_video() {
  local ok=1 job a b out font=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
  for job in ${CLIPS//,/ }; do
    a=$VID/main_r1_$job.mp4; b=$VID/wt_all_$job.mp4; out=$VID/scheduler_main_vs_candidate_$job.mp4
    if [[ ! -s $a || ! -s $b ]]; then echo "FAIL video $job: missing $a or $b (run g1 and g5)"; ok=0; continue; fi
    ffmpeg -hide_banner -loglevel error -y -i "$a" -i "$b" -filter_complex \
      "[0:v]format=rgb24,split[a0][a1];[1:v]format=rgb24,split[b0][b1];[a1][b1]blend=all_mode=difference,lutrgb=r=val*8:g=val*8:b=val*8[d];\
[a0]drawtext=fontfile=$font:text='A  main today':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[la];\
[b0]drawtext=fontfile=$font:text='B  worktree + flags':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[lb];\
[d]drawtext=fontfile=$font:text='8x |A-B|':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[ld];\
[la][lb][ld]hstack=inputs=3" -c:v libx264 -crf 12 -pix_fmt yuv420p "$out" || { ok=0; continue; }
    echo "  built $out"
  done
  $PY - "$D" "$VID" "$ok" <<'EOF'
import json, sys, time, pathlib
d, vid, ok = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
files = sorted(str(p) for p in pathlib.Path(vid).glob("scheduler_main_vs_candidate_*.mp4"))
json.dump({"gate": "comparison_video", "verdict": "PASS" if ok and files else "FAIL", "files": files,
           "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{d}/gate_comparison_video.json", "w"), indent=1)
print(("PASS" if ok and files else "FAIL") + f" comparison_video: {len(files)} clips in {vid}")
EOF
}

step_summary() { $GATE summary; }

STEPS=("$@")
[[ ${#STEPS[@]} -eq 0 ]] && STEPS=(cpu g1 g2 g3 g4 g5 g6 t1 t2 video summary)
for step in "${STEPS[@]}"; do
  echo "##### step $step  $(date -u +%H:%M:%S)"
  case "$step" in
    cpu|g1|g2|g3|g4|g5|g6|t1|t2|video|summary) "step_$step" ;;
    *) echo "unknown step $step (cpu g1 g2 g3 g4 g5 g6 t1 t2 video summary)"; exit 2 ;;
  esac
  echo "##### step $step rc=$?  $(date -u +%H:%M:%S)"
  if [[ $step == g1 || $step == g2 ]]; then
    gate_file=$D/gate_main_in_process.json
    [[ $step == g2 ]] && gate_file=$D/gate_main_fresh_process.json
    if ! grep -q '"verdict": "PASS"' "$gate_file" 2>/dev/null; then
      echo "STOP: the main baseline does not reproduce ($gate_file); candidate gates would be meaningless."
      exit 1
    fi
  fi
done
