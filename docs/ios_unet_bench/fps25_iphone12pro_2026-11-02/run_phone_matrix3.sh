#!/bin/zsh
# run_phone_matrix3.sh [max wait minutes]
# Third iPhone 12 Pro matrix: completes matrix2's missing mirror of the shipped run (it hung when SoulX Studio
# opened mid-launch) and tries a third Neural Engine worker. Order A D E E D A:
#   A shipped UNet, 1 worker      D V2-folded UNet, 2 workers      E V2-folded UNet, 3 workers
# Waits for the phone (connected, unlocked, no SoulX app), cools 300 s, then runs each step under a watchdog: if a
# SoulX app appears or a step runs past 150 s, the launcher is stopped, the step is marked aborted, and it is retried
# once the phone is free again (twice at most). The app is already installed with both UNets compiled.
# Summary: $SCRATCH/phone_m3_summary.txt.
set -u
UDID=00008101-00123DA43A51003A
SCRATCH=/private/tmp/claude-501/-Users-ahmadsmacair-code-lingua-lingua/f0f23d41-3e43-42a1-916d-19f56c854684/scratchpad
RUN=$SCRATCH/run_phone.sh
BASE=unet_anec0_int8w_b1_chunked; CAND=unet_anev2fc0_int8w_b1_chunked
WAIT_MIN=${1:-600}
SUM=$SCRATCH/phone_m3_summary.txt; STEPS=$SCRATCH/phone_m3_steps.log
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
    print("[%s %s] gen %d/%d = %.2f fps, shown %d, infer %.1f ms, compose %.1f ms, decode %.1f ms, skipped %d, late %d, stalls %d, lead %d, workers %s, unet %s, thermal_after %s" % (
        n, k, d["generated"], d["frames"], d["generated_fps"], d["shown"], d["stage_infer_ms"], d["stage_compose_ms"],
        d["decode_ms"], d["skipped"], d["dropped_late"], d["stalls"], d["lead_frames"], d.get("infer_workers"), d["unet"],
        d.get("thermal_after", "?")))
' "$2" "$3"
}

wait_ready
echo "== $(date '+%H:%M:%S') SoulX closed and phone unlocked; cooling 300 s" | tee -a "$SUM"
sleep 300
wait_ready
echo "== $(date '+%H:%M:%S') phone ready" | tee -a "$SUM"

typeset -A ARGS
ARGS[A]="--unet $BASE"
ARGS[D]="--unet $CAND --infer-workers 2"
ARGS[E]="--unet $CAND --infer-workers 3"
n=0
for k in A D E E D A; do
  n=$((n + 1))
  for attempt in 1 2 3; do
    wait_ready
    label=m3_${n}_$k
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
    wait_ready; sleep 120
  done
  sleep 45
done
echo "== $(date '+%H:%M:%S') done" | tee -a "$SUM"
