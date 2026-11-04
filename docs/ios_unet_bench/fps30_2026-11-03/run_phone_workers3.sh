#!/bin/zsh
# run_phone_workers3.sh [max wait minutes]
# iPhone 12 Pro: does a third Neural Engine worker add frames on the A14? (On the Mac the ANE is saturated with two.)
# Also re-measures the shipped single-worker run at the same temperature, for round 1's gain. Order A D E E D A, the
# Lipsync tab's 17.8 s clip each:
#   A shipped UNet, 1 worker      D folded V2 UNet, 2 workers (best so far)      E folded V2 UNet, 3 workers
# Waits for the phone (connected, unlocked, no SoulX app), cools 300 s, runs both UNets for 20 frames (compiles the
# shipped one, which the 2026-11-03 reinstall left uncompiled; reports the thermal state; stops if the UNet is not on
# the Neural Engine), then runs each step under the SoulX watchdog (a SoulX app appearing, or a step past 150 s, stops
# it and retries once the phone is free, twice at most), and ends with a 5-frame run for the thermal state after.
# The current harness build must already be installed. Summary: $SCRATCH/phone_w3_summary.txt.
set -u
UDID=00008101-00123DA43A51003A
SCRATCH=/private/tmp/claude-501/-Users-ahmadsmacair-code-lingua-lingua/f0f23d41-3e43-42a1-916d-19f56c854684/scratchpad
RUN=$SCRATCH/run_phone.sh
BASE=unet_anec0_int8w_b1_chunked; CAND=unet_anev2fc0_int8w_b1_chunked
WAIT_MIN=${1:-600}
SUM=$SCRATCH/phone_w3_summary.txt; STEPS=$SCRATCH/phone_w3_steps.log
: > "$SUM"; : > "$STEPS"

soulx() { xcrun devicectl device info processes --device "$UDID" 2>/dev/null | grep -qi soulx; }
ready() {
  xcrun devicectl list devices 2>/dev/null | grep "$UDID" | grep -qv unavailable || return 1
  xcrun devicectl device info lockState --device "$UDID" 2>/dev/null | grep -q "passcodeRequired: false" || return 1
  ! soulx
}
deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
wait_ready() {
  until ready; do
    (( $(date +%s) > deadline )) && { echo "phone not ready within $WAIT_MIN min" | tee -a "$SUM"; exit 3; }
    sleep 20
  done
}
summarize() {  # <log> <n> <k>
  grep -h "LIPSYNC_RESULT" "$1" | head -1 | sed 's/^LIPSYNC_RESULT //' | python3 -I -c '
import sys, json
n, k = sys.argv[1], sys.argv[2]
for l in sys.stdin:
    d = json.loads(l)
    print("[%s %s] gen %d/%d = %.2f fps, shown %d, ANE stage %.1f ms, compose %.1f ms, decode %.1f ms, skipped %d, late %d, stalls %d, lead %d, workers %s, unet %s" % (
        n, k, d["generated"], d["frames"], d["generated_fps"], d["shown"], d["stage_infer_ms"], d["stage_compose_ms"],
        d["decode_ms"], d["skipped"], d["dropped_late"], d["stalls"], d["lead_frames"], d.get("infer_workers"), d["unet"]))
' "$2" "$3"
}
BENCH=(--autorun --sustained 0 --chain 0 --no-gpu --no-b4 --no-whisper --no-plan --no-reload --units CPU_AND_NE --exit)

wait_ready
echo "== $(date '+%H:%M:%S') SoulX closed and phone unlocked; cooling 300 s" | tee -a "$SUM"
sleep 300
wait_ready
echo "== $(date '+%H:%M:%S') phone ready" | tee -a "$SUM"

"$RUN" w3_compile -- "${BENCH[@]}" --frames 20 --models "$BASE,$CAND" >>"$STEPS" 2>&1
grep -hE "device:|ms/frame mean" "$SCRATCH/phone_w3_compile.log" | grep -v '^ *"' | sed 's/^/[compile] /' >> "$SUM"
# A phone that lost the Neural Engine runs the UNet on the CPU at ~1 s/frame; healthy is 53-72 ms. Stop rather than
# measure that.
for ms in $(grep -hoE "[0-9.]+ ms/frame mean" "$SCRATCH/phone_w3_compile.log" | cut -d' ' -f1) none; do
  [[ "$ms" == none ]] && break
  (( ${ms%.*} > 150 )) && { echo "== Neural Engine looks unavailable (UNet $ms ms/frame): restart the phone" | tee -a "$SUM"; exit 4; }
done
grep -q "ms/frame mean" "$SCRATCH/phone_w3_compile.log" || { echo "== compile step gave no result" | tee -a "$SUM"; exit 4; }
sleep 120

typeset -A ARGS
ARGS[A]="--unet $BASE --infer-workers 1"
ARGS[D]="--unet $CAND --infer-workers 2"
ARGS[E]="--unet $CAND --infer-workers 3"
n=0
for k in A D E E D A; do
  n=$((n + 1))
  for attempt in 1 2 3; do
    wait_ready
    label=w3_${n}_$k
    "$RUN" $label -- --lipsync --exit --silent ${=ARGS[$k]} >>"$STEPS" 2>&1 &
    pid=$!
    start=$(date +%s); aborted=""
    while kill -0 $pid 2>/dev/null; do
      sleep 10
      if soulx; then aborted="SoulX app appeared"; elif (( $(date +%s) - start > 150 )); then aborted="ran past 150 s"; fi
      if [[ -n "$aborted" ]]; then
        pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null
        pkill -f "devicectl device process launch.*com.lingua.musetalkbench" 2>/dev/null
        break
      fi
    done
    wait $pid 2>/dev/null
    if [[ -z "$aborted" ]] && grep -q LIPSYNC_RESULT "$SCRATCH/phone_$label.log"; then
      summarize "$SCRATCH/phone_$label.log" $n $k >> "$SUM"
      break
    fi
    echo "[$n $k] attempt $attempt aborted at $(date '+%H:%M:%S'): ${aborted:-no result}" >> "$SUM"
    grep -hE "rror|jetsam|memory" "$SCRATCH/phone_$label.log" | tail -3 | sed "s/^/[$n $k] /" >> "$SUM"
    wait_ready; sleep 120
  done
  sleep 45
done

wait_ready
"$RUN" w3_thermal -- "${BENCH[@]}" --frames 5 --models "$CAND" >>"$STEPS" 2>&1
grep -hE "device:" "$SCRATCH/phone_w3_thermal.log" | grep -v '^ *"' | sed 's/^/[after] /' >> "$SUM"
echo "== $(date '+%H:%M:%S') done" | tee -a "$SUM"
