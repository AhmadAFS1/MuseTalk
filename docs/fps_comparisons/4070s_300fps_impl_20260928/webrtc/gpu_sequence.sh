#!/usr/bin/env bash
# GPU validation sequence for the WebRTC / API / load-harness items (plan 0.4, 0.5, 1.5, 1.7,
# 1.8, 1.11). NOT run by the CPU-only workflow that wrote it. Every GPU step is ONE box_guard
# lease that starts the server (experiments/throughput300_candidate/run_candidate_api.sh, port
# 8300), drives it with load_test_webrtc_v2.py (clients pinned to physical cores 12-15 =
# CPUs 12-15,28-31; server pinned to 0-11,16-27), stops it, and prints PASS/FAIL lines.
#
# Usage: gpu_sequence.sh <step> [<step> ...]      (steps run in the order given)
#        gpu_sequence.sh all                      (= S0 X1 LB LC1 LC2 ENC SUM)
# Stop your live server first (D3); the lease also honours /workspace/.gpu_lease.pause.
#
# Step  What                                              Lease / RAM gate         Runtime
# ----  ------------------------------------------------  -----------------------  --------
# S0    Loopback smoke, candidate overlay + WEBRTC_HANDOFF_VERIFY=1, N=3, 2 turns each,
#       standard avatar.                                  --min-avail-gb 14        ~6 min
#       PASS if every turn completes, output_fps 20, fresh >= 0.99, 0 stalls, every stream
#       idle-cache-backed, non-blocking handoff with 0 order/content/I420 mismatches, scheduler
#       callback <= 2 ms/batch, VP8 encoder = native.  -> S0_smoke.json, S0/…
# X1    E0 exactness A/B of the pre-encoder I420 frames: baseline server (overlay none),
#       candidate server, baseline again; each N=1, bob pose session, WAVs turn_one + turn_two,
#       idle phase pinned (WEBRTC_TEST_IDLE_SYNC_SOURCE_FRAME=0), tap WEBRTC_PREENCODE_SHA_DIR.
#                                                       --min-avail-gb 14        ~14 min
#       First line: PASS if base == base2 (the capture reproduces; otherwise the A/B is
#       inconclusive). Second line: PASS if candidate == base: same turns, frame counts and
#       SHA-256 of every frame (G-EXACT for 1.5 / 1.7 / 1.11).
#       -> X1_reproducibility.json, X1_exactness.json
# LB    Baseline load levels N=1,5,10, --chain (every stream speaks the whole window),
#       >= 180 s steady state each.                      --min-avail-gb 14        ~18 min
#       Reported per level: aggregate fresh fps, per-stream fresh fraction, max held run, stall
#       seconds, server send max/avg interval, first-frame p50/p95, speakers, server
#       RSS/threads/CPU/VRAM, client loop lag.  Plan pass criteria (fresh >= 99.5%, held run
#       <= 2, 0 stall s, send avg 0.050+-0.001, max <= 0.100, output_fps 20) give PASS/FAIL;
#       the baseline is expected to FAIL at 10 (it was 68-72% fresh at 10 streams).
#       -> LB/baseline_n{01,05,10}.json, LB/baseline_summary.json
# LC1   Candidate levels N=1,5,10 (same method).        --min-avail-gb 14        ~18 min
# LC2   Candidate levels N=12,15,20 (same method).      --min-avail-gb 14        ~20 min
#       Target: PASS at 15 with aggregate fresh fps >= 297 (plan section 6.4); 20 = stretch.
#       -> LC/candidate_n*.json, LC/candidate_*_summary.json
# ENC   NVENC leg of the G-ENC encoder-quality test (h264_nvenc vs today's libx264 at 2.5 Mbps,
#       PSNR/SSIM on real avatar frames).               --min-avail-gb 6         ~2 min
#       -> ENC_nvenc_quality.json (PASS within 0.2 dB of today's H.264)
# SUM   CPU only (no lease): baseline vs candidate table from LB/LC JSONs.  -> SUM_levels.json
#
# Overrides: AVATARS (comma list, default two standard avatars), LEVELS_LB / LEVELS_LC1 /
# LEVELS_LC2, CHAIN_S (default 220), PORT (8300).
set -uo pipefail
R=/workspace/MuseTalk-perf300
D=$R/docs/fps_comparisons/4070s_300fps_impl_20260928/webrtc
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
LAUNCH=$R/experiments/throughput300_candidate/run_candidate_api.sh
OVERLAY=$R/experiments/throughput300_candidate/musetalk_300fps.env
CORPUS=$R/experiments/throughput300_candidate/audio_corpus
BOB=/workspace/experiments/chinese_bob_webrtc_20260927
PORT=${PORT:-8300}
AVATARS=${AVATARS:-japanese_realtime_talking_7d94520b7f,latina_guided_20260925_talking_84c5bc80b8_fh1}
CHAIN_S=${CHAIN_S:-220}
LEVELS_LB=${LEVELS_LB:-1,5,10}
LEVELS_LC1=${LEVELS_LC1:-1,5,10}
LEVELS_LC2=${LEVELS_LC2:-12,15,20}
G="$R/scripts/box_guard.sh run --wait-min 60 --kill-below-gb 4"
cd "$R"

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$SERVER_PID" 2>/dev/null
  fi
  SERVER_PID=""
}
trap stop_server EXIT

# start_server <overlay-path|none> <log> [extra VAR=value ...]; waits for /health (<= 15 min)
start_server() {
  local overlay=$1 log=$2; shift 2
  env "$@" MUSETALK_300FPS_OVERLAY="$overlay" MUSETALK_API_PORT="$PORT" MUSETALK_CANDIDATE_LOG="$log" \
    bash "$LAUNCH" &
  SERVER_PID=$!
  local t0=$SECONDS
  while (( SECONDS - t0 < 900 )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "FAIL server exited during startup (see $log)"; return 1; fi
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "server ready on :$PORT after $((SECONDS - t0)) s (overlay=$overlay, log=$log)"
      return 0
    fi
    sleep 5
  done
  echo "FAIL server not healthy after 900 s (see $log)"; return 1
}

harness() {  # harness <label> <out-dir> [args...]
  local label=$1 out=$2; shift 2
  $PY load_test_webrtc_v2.py --base-url "http://127.0.0.1:$PORT" --label "$label" --out-dir "$out" \
    --audio-dir "$CORPUS" --avatar-ids "$AVATARS" --ignore-ice-servers --prebuffer-seconds 0.5 "$@"
}

inner_S0() {
  mkdir -p "$D/S0"
  start_server "$OVERLAY" "$D/S0/server.log" WEBRTC_HANDOFF_VERIFY=1 || return 1
  harness smoke "$D/S0" --levels 3 --turns 2 --min-steady-s 10 --warmup-s 5 --keep-sessions
  $PY "$D/check_smoke.py" "$D/S0/smoke_n03.json" --expect-vp8 native --out "$D/S0_smoke.json"
}

inner_X1() {
  mkdir -p "$D/X1"
  rm -rf "$D/X1/tap_base" "$D/X1/tap_cand" "$D/X1/tap_base2"
  local wavs="$BOB/audio/turn_one.wav,$BOB/audio/turn_two.wav"
  local common=(--avatar-ids chinese_bob_pink_bedroom_idle_d4b06da317 --pose-set-file "$BOB/session-pose-set.json"
                --levels 1 --wav-list "$wavs" --turn-gap-s 3 --min-steady-s 1 --warmup-s 1)
  start_server none "$D/X1/server_base.log" WEBRTC_PREENCODE_SHA_DIR="$D/X1/tap_base" \
      WEBRTC_TEST_IDLE_SYNC_SOURCE_FRAME=0 || return 1
  harness x1_base "$D/X1" "${common[@]}"
  stop_server; sleep 10
  start_server "$OVERLAY" "$D/X1/server_cand.log" WEBRTC_PREENCODE_SHA_DIR="$D/X1/tap_cand" \
      WEBRTC_TEST_IDLE_SYNC_SOURCE_FRAME=0 || return 1
  harness x1_cand "$D/X1" "${common[@]}"
  stop_server; sleep 10
  # Second baseline run: proves the capture itself reproduces (a candidate mismatch is only
  # meaningful if base == base2).
  start_server none "$D/X1/server_base2.log" WEBRTC_PREENCODE_SHA_DIR="$D/X1/tap_base2" \
      WEBRTC_TEST_IDLE_SYNC_SOURCE_FRAME=0 || return 1
  harness x1_base2 "$D/X1" "${common[@]}"
  stop_server
  $PY "$D/compare_preencode_sha.py" "$D/X1/tap_base" "$D/X1/tap_base2" --out "$D/X1_reproducibility.json"
  $PY "$D/compare_preencode_sha.py" "$D/X1/tap_base" "$D/X1/tap_cand" --out "$D/X1_exactness.json"
}

inner_levels() {  # inner_levels <overlay|none> <label> <levels> <dir>
  local overlay=$1 label=$2 levels=$3 dir=$4
  mkdir -p "$dir"
  start_server "$overlay" "$dir/server_${label}.log" || return 1
  harness "$label" "$dir" --levels "$levels" --chain --chain-seconds "$CHAIN_S" --min-steady-s 180 \
    --settle-s 5 --cooldown-s 30
}

inner_ENC() {
  $PY "$D/test_encoder_quality.py" --configs h264_nvenc --out "$D/ENC_nvenc_quality.json" \
    | tee "$D/ENC_nvenc_quality.log" | grep -E "^(PASS|FAIL)"
}

summarize() {
  $PY - "$D" <<'EOF'
import glob, json, sys
from pathlib import Path
d = Path(sys.argv[1]); rows = []
for path in sorted(glob.glob(str(d / "L*" / "*_n[0-9][0-9].json"))):
    s = json.load(open(path)); a = s["aggregate"]; srv = s["server"]
    rows.append({"label": s["label"], "N": s["level"], "verdict": s["verdict"],
                 "steady_s": s["window"]["steady_state_s"], "agg_fresh_fps": a.get("aggregate_fresh_fps_mean"),
                 "agg_fresh_fps_min_1s": a.get("aggregate_fresh_fps_min_1s"), "generated_fps": a.get("generated_fps_server"),
                 "fresh_frac_min": a.get("fresh_fraction_min"), "held_run_max": a.get("max_held_run"),
                 "stall_s_max": a.get("stall_seconds_max"), "send_max_s": a.get("send_interval_max_s"),
                 "first_frame_p95_s": a.get("first_frame_p95_s"), "speakers_min": a.get("speakers_min"),
                 "callback_ms": a.get("callback_ms_per_batch"), "server_cpu": srv.get("server_cpu_cores_mean"),
                 "rss_max_mb": srv.get("server_rss_max_mb"), "threads_max": srv.get("server_threads_max"),
                 "vram_max_mb": srv.get("server_vram_max_mb"), "gpu_util": srv.get("gpu_util_mean"),
                 "failed": [k for k, v in s["checks"].items() if not v],
                 "invalid": [k for k, v in s["validity"].items() if not v]})
(d / "SUM_levels.json").write_text(json.dumps(rows, indent=2))
for r in rows:
    print(f"{r['verdict']:7s} {r['label']:10s} N={r['N']:2d} fresh_fps={r['agg_fresh_fps']} gen_fps={r['generated_fps']} "
          f"fresh_min={r['fresh_frac_min']} held_run={r['held_run_max']} stall={r['stall_s_max']} send_max={r['send_max_s']} "
          f"ff_p95={r['first_frame_p95_s']} cb_ms={r['callback_ms']} cpu={r['server_cpu']} rss={r['rss_max_mb']} "
          f"thr={r['threads_max']} vram={r['vram_max_mb']} failed={r['failed']} invalid={r['invalid']}")
c15 = [r for r in rows if r["label"].startswith("candidate") and r["N"] == 15]
if c15:
    r = c15[0]
    ok = r["verdict"] == "PASS" and (r["agg_fresh_fps"] or 0) >= 297
    print(f"{'PASS' if ok else 'FAIL'} 300fps/15-stream target: candidate N=15 verdict={r['verdict']} "
          f"agg_fresh_fps={r['agg_fresh_fps']} (need >= 297)")
EOF
}

# ------------------------------------------------------------------------------------ dispatch
if [[ "${1:-}" == _inner ]]; then
  case "$2" in
    S0) inner_S0 ;;
    X1) inner_X1 ;;
    LB) inner_levels none baseline "$LEVELS_LB" "$D/LB" ;;
    LC1) inner_levels "$OVERLAY" candidate_a "$LEVELS_LC1" "$D/LC" ;;
    LC2) inner_levels "$OVERLAY" candidate_b "$LEVELS_LC2" "$D/LC" ;;
    ENC) inner_ENC ;;
    *) echo "unknown inner step $2"; exit 2 ;;
  esac
  exit $?
fi

steps=("$@")
[[ ${#steps[@]} -eq 0 ]] && { sed -n '2,45p' "$0"; exit 2; }
[[ "${steps[0]}" == all ]] && steps=(S0 X1 LB LC1 LC2 ENC SUM)
for step in "${steps[@]}"; do
  echo "=== step $step $(date -u +%H:%M:%S)"
  case "$step" in
    S0)  $G --min-avail-gb 14 --label webrtc_S0_smoke -- bash "$0" _inner S0 2>&1 | tee "$D/S0.log" ;;
    X1)  $G --min-avail-gb 14 --label webrtc_X1_exactness -- bash "$0" _inner X1 2>&1 | tee "$D/X1.log" ;;
    LB)  $G --min-avail-gb 14 --label webrtc_LB_baseline -- bash "$0" _inner LB 2>&1 | tee "$D/LB.log" ;;
    LC1) $G --min-avail-gb 14 --label webrtc_LC1_candidate -- bash "$0" _inner LC1 2>&1 | tee "$D/LC1.log" ;;
    LC2) $G --min-avail-gb 14 --label webrtc_LC2_candidate -- bash "$0" _inner LC2 2>&1 | tee "$D/LC2.log" ;;
    ENC) $G --min-avail-gb 6 --label webrtc_ENC_nvenc -- bash "$0" _inner ENC 2>&1 | tee "$D/ENC.log" ;;
    SUM) summarize 2>&1 | tee "$D/SUM.log" ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  echo "=== step $step rc=${PIPESTATUS[0]} $(date -u +%H:%M:%S)"
done
