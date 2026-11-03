#!/bin/zsh
# run_phone_matrix2.sh [max wait minutes]
# Second iPhone 12 Pro matrix for the 25 fps levers (the first showed the A14 GPU takes 67 ms per TAESD decode).
# Waits for the phone (connected, unlocked, no SoulX app), cools 300 s, installs the current harness build, compiles,
# then plays the bundled 17.8 s clip in the Lipsync tab in the order A B C D D C B A:
#   A shipped UNet, TAESD in the ANE stage, 1 worker        B the same with 2 workers
#   C V2-folded UNet, TAESD on the CPU in the compose stage, 2 workers
#   D V2-folded UNet, TAESD in the ANE stage, 2 workers
# Summary: $SCRATCH/phone_m2_summary.txt.
set -u
UDID=00008101-00123DA43A51003A
SCRATCH=/private/tmp/claude-501/-Users-ahmadsmacair-code-lingua-lingua/f0f23d41-3e43-42a1-916d-19f56c854684/scratchpad
RUN=$SCRATCH/run_phone.sh
BASE=unet_anec0_int8w_b1_chunked; CAND=unet_anev2fc0_int8w_b1_chunked
WAIT_MIN=${1:-600}
SUM=$SCRATCH/phone_m2_summary.txt; STEPS=$SCRATCH/phone_m2_steps.log
: > "$SUM"; : > "$STEPS"

ready() {
  xcrun devicectl list devices 2>/dev/null | grep "$UDID" | grep -qv unavailable || return 1
  xcrun devicectl device info lockState --device "$UDID" 2>/dev/null | grep -q "passcodeRequired: false" || return 1
  ! xcrun devicectl device info processes --device "$UDID" 2>/dev/null | grep -qi soulx
}
deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
wait_ready() {
  until ready; do
    (( $(date +%s) > deadline )) && { echo "phone not ready within $WAIT_MIN min" | tee -a "$SUM"; exit 3; }
    sleep 20
  done
}
wait_ready
echo "== $(date '+%H:%M:%S') SoulX closed and phone unlocked; cooling 300 s" | tee -a "$SUM"
sleep 300
wait_ready
echo "== $(date '+%H:%M:%S') phone ready" | tee -a "$SUM"

BENCH=(--autorun --frames 20 --sustained 0 --chain 0 --no-gpu --no-b4 --no-whisper --no-plan --no-reload --units CPU_AND_NE --exit)
"$RUN" m2_compile --install -- "${BENCH[@]}" --models "$BASE,$CAND" >>"$STEPS" 2>&1
grep -hE "thermal|ms/frame mean" "$SCRATCH/phone_m2_compile.log" | head -3 | sed 's/^/[compile] /' >> "$SUM"
sleep 120

typeset -A ARGS
ARGS[A]="--unet $BASE"
ARGS[B]="--unet $BASE --infer-workers 2"
ARGS[C]="--unet $CAND --defer-decode --taesd-units CPU_ONLY --infer-workers 2"
ARGS[D]="--unet $CAND --infer-workers 2"
n=0
for k in A B C D D C B A; do
  n=$((n + 1))
  wait_ready
  "$RUN" m2_${n}_$k -- --lipsync --exit --silent ${=ARGS[$k]} >>"$STEPS" 2>&1
  grep -h "LIPSYNC_RESULT" "$SCRATCH/phone_m2_${n}_$k.log" | head -1 | sed 's/^LIPSYNC_RESULT //' | python3 -I -c '
import sys, json
k, n = sys.argv[1], sys.argv[2]
for l in sys.stdin:
    d = json.loads(l)
    print(f"[{n} {k}] gen {d[\"generated\"]}/{d[\"frames\"]} = {d[\"generated_fps\"]:.2f} fps, shown {d[\"shown\"]}, infer {d[\"stage_infer_ms\"]:.1f} ms, "
          f"compose {d[\"stage_compose_ms\"]:.1f} ms, decode {d[\"decode_ms\"]:.1f} ms, skipped {d[\"skipped\"]}, late {d[\"dropped_late\"]}, "
          f"stalls {d[\"stalls\"]}, lead {d[\"lead_frames\"]}, workers {d.get(\"infer_workers\")}, unet {d[\"unet\"]}")
' "$k" "$n" >> "$SUM"
  sleep 45
done
echo "== $(date '+%H:%M:%S') done" | tee -a "$SUM"
