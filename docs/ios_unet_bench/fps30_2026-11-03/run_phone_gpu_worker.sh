#!/bin/zsh
# run_phone_gpu_worker.sh [max wait minutes]
# iPhone 12 Pro: does a GPU worker next to the two Neural Engine workers add frames on the A14, as it does on the
# Mac (1.36-1.87x there)? Order D H H D, the Lipsync tab's 17.8 s clip each:
#   D folded V2 UNet, 2 Neural Engine workers (2026-11-02's best on the phone, 18.6 fps hot)
#   H the same + 1 GPU worker running the same chunks on the GPU (no new model files on the phone)
# Waits for the phone (connected, unlocked, no SoulX app), cools 300 s, installs the current harness build
# (App/DerivedData, built with the GPU worker), compiles both UNets on the Neural Engine, then runs each step under a
# watchdog: a SoulX app appearing, or a step past 300 s (H warms the GPU model up first: 17 s on the Mac), stops it
# and retries once the phone is free (twice at most). Summary: $SCRATCH/phone_gpu_summary.txt.
set -u
UDID=00008101-00123DA43A51003A
SCRATCH=/private/tmp/claude-501/-Users-ahmadsmacair-code-lingua-lingua/f0f23d41-3e43-42a1-916d-19f56c854684/scratchpad
RUN=$SCRATCH/run_phone.sh
CAND=unet_anev2fc0_int8w_b1_chunked
WAIT_MIN=${1:-600}
SUM=$SCRATCH/phone_gpu_summary.txt; STEPS=$SCRATCH/phone_gpu_steps.log
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
    print("[%s %s] gen %d/%d = %.2f fps, shown %d, ANE stage %.1f ms, GPU %d frames at %.1f ms, compose %.1f ms, decode %.1f ms, skipped %d, late %d, stalls %d, lead %d, workers %s+%s gpu" % (
        n, k, d["generated"], d["frames"], d["generated_fps"], d["shown"], d["stage_infer_ms"], d.get("gpu_generated", 0),
        d.get("gpu_ms", 0), d["stage_compose_ms"], d["decode_ms"], d["skipped"], d["dropped_late"], d["stalls"],
        d["lead_frames"], d.get("infer_workers"), d.get("gpu_workers", 0)))
' "$2" "$3"
}

wait_ready
echo "== $(date '+%H:%M:%S') SoulX closed and phone unlocked; cooling 300 s" | tee -a "$SUM"
sleep 300
wait_ready
echo "== $(date '+%H:%M:%S') phone ready" | tee -a "$SUM"

BENCH=(--autorun --frames 20 --sustained 0 --chain 0 --no-gpu --no-b4 --no-whisper --no-plan --no-reload --units CPU_AND_NE --exit)
"$RUN" gpu_compile --install -- "${BENCH[@]}" --models "$CAND" >>"$STEPS" 2>&1
grep -hE "thermal|ms/frame mean" "$SCRATCH/phone_gpu_compile.log" | head -3 | sed 's/^/[compile] /' >> "$SUM"
# A phone that lost the Neural Engine (seen after a SoulX interruption until a restart) runs the UNet on the CPU at
# ~1 s/frame; healthy is 55-66 ms. Stop rather than measure that.
ms=$(grep -hoE "[0-9.]+ ms/frame mean" "$SCRATCH/phone_gpu_compile.log" | head -1 | cut -d' ' -f1)
if [[ -z "$ms" ]] || (( ${ms%.*} > 150 )); then
  echo "== Neural Engine looks unavailable (UNet ${ms:-no result} ms/frame): restart the phone" | tee -a "$SUM"; exit 4
fi
sleep 120

typeset -A ARGS
ARGS[D]="--unet $CAND --infer-workers 2"
ARGS[H]="--unet $CAND --infer-workers 2 --gpu-unet $CAND"
n=0
for k in D H H D; do
  n=$((n + 1))
  for attempt in 1 2 3; do
    wait_ready
    label=gpu_${n}_$k
    "$RUN" $label -- --lipsync --exit --silent ${=ARGS[$k]} >>"$STEPS" 2>&1 &
    pid=$!
    start=$(date +%s); aborted=""
    while kill -0 $pid 2>/dev/null; do
      sleep 10
      if soulx; then aborted="SoulX app appeared"; elif (( $(date +%s) - start > 300 )); then aborted="ran past 300 s"; fi
      if [[ -n "$aborted" ]]; then
        pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null
        pkill -f "devicectl device process launch.*com.lingua.musetalkbench" 2>/dev/null
        break
      fi
    done
    wait $pid 2>/dev/null
    if [[ -z "$aborted" ]] && grep -q LIPSYNC_RESULT "$SCRATCH/phone_$label.log"; then
      summarize "$SCRATCH/phone_$label.log" $n $k >> "$SUM"
      grep -hE "GPU worker warm-up|memory pressure|footprint" "$SCRATCH/phone_$label.log" | head -3 | sed "s/^/[$n $k] /" >> "$SUM"
      break
    fi
    echo "[$n $k] attempt $attempt aborted at $(date '+%H:%M:%S'): ${aborted:-no result}" >> "$SUM"
    grep -hE "GPU|rror|jetsam|memory" "$SCRATCH/phone_$label.log" | tail -3 | sed "s/^/[$n $k] /" >> "$SUM"
    wait_ready; sleep 120
  done
  sleep 45
done
echo "== $(date '+%H:%M:%S') done" | tee -a "$SUM"
