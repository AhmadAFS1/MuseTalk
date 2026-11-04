#!/bin/zsh
# run_phone_small_unets.sh <out dir> [order]   (env NO_INSTALL=1 skips the install)
# iPhone 12 Pro: the speed of the untrained smaller UNets (scripts/ios/export_small_unets.py; results in this folder) on the A14's Neural
# Engine, one bench launch per model (UNet alone, 50 calls after an 8-frame gate), in a mirrored order with today's
# phone UNet (Apple layout, folded V2 chunks) as the anchor:
#   A unet_anev2fc0_int8w_b1_chunked   P unet_int8w_b1 (plain full)   B bk_base   S bk_small   T bk_tiny
# The plain full model does not load as one program on the 12 Pro (memory pressure critical at 2.46 GB, the app
# killed), so the default order leaves it out and A anchors the ratios.
# Installs the current harness build first (App/DerivedData, with the four plain UNets in HarnessBundle). Every launch
# runs under the SoulX watchdog (a SoulX app appearing, or a launch past 300 s, stops it; retried twice), and the
# script stops if a UNet frame takes over 150 ms (not on the Neural Engine; a nearly full disk does that).
set -u
OUT=$1; ORDER=(${=${2:-A B S T T S B A}}); mkdir -p "$OUT"; SUM=$OUT/summary.txt; : > "$SUM"
UDID=00008101-00123DA43A51003A; BUNDLE=com.lingua.musetalkbench
APP=/Users/ahmadsmacair/code/musetalk-ios-bench/ios/MuseTalkBench/App/DerivedData/Build/Products/Release-iphoneos/MuseTalkBench.app
typeset -A M
M[A]=unet_anev2fc0_int8w_b1_chunked; M[P]=unet_int8w_b1; M[B]=unet_bk_base_int8w_b1; M[S]=unet_bk_small_int8w_b1; M[T]=unet_bk_tiny_int8w_b1
BENCH=(--autorun --frames 50 --sustained 0 --chain 0 --no-gpu --no-b4 --no-whisper --no-plan --no-reload --units CPU_AND_NE --accuracy-frames 8 --exit)

soulx() { xcrun devicectl device info processes --device "$UDID" 2>/dev/null | grep -qi soulx; }
ready() {
  xcrun devicectl list devices 2>/dev/null | grep "$UDID" | grep -qv unavailable || return 1
  xcrun devicectl device info lockState --device "$UDID" 2>/dev/null | grep -q "passcodeRequired: false" || return 1
  ! soulx
}
wait_ready() { local dl=$(( $(date +%s) + 3600 )); until ready; do (( $(date +%s) > dl )) && { echo "== phone not ready" >> "$SUM"; exit 3; }; sleep 20; done; }
guarded() {  # <log> <app args...>
  local log=$1; shift
  print -r -- "== $(date '+%H:%M:%S') $*" > "$log"
  xcrun devicectl device process launch --device "$UDID" --console --terminate-existing "$BUNDLE" "$@" >> "$log" 2>&1 &
  local pid=$! start=$(date +%s) why=""
  while kill -0 $pid 2>/dev/null; do
    sleep 5
    if soulx; then why="SoulX app appeared"; elif (( $(date +%s) - start > 300 )); then why="ran past 300 s"; fi
    if [[ -n $why ]]; then
      kill $pid 2>/dev/null; wait $pid 2>/dev/null
      echo "== $(date '+%H:%M:%S') stopped ${log:t}: $why" >> "$SUM"; return 1
    fi
  done
  wait $pid 2>/dev/null; return 0
}

wait_ready
if [[ -z ${NO_INSTALL:-} ]]; then
xcrun devicectl device install app --device "$UDID" "$APP" > "$OUT/install.log" 2>&1 || { echo "== install failed" >> "$SUM"; exit 5; }
echo "== $(date '+%H:%M:%S') installed" >> "$SUM"
fi
n=0
for k in $ORDER; do
  n=$((n + 1))
  for attempt in 1 2 3; do
    wait_ready
    log=$OUT/${n}_${k}_a$attempt.log
    if guarded "$log" "${BENCH[@]}" --models "${M[$k]}"; then
      line=$(grep -h '^\[bench\]' "$log" | grep -E "ms/frame mean" | head -1)
      if [[ -n $line ]]; then
        th=$(grep -h '^\[bench\]' "$log" | grep -oE 'thermal (state -> )?[a-z]+' | tail -1 | awk '{print $NF}')
        disk=$(grep -h '^\[bench\] free disk' "$log" | head -1 | grep -oE '[0-9]+' | head -1)
        load=$(grep -h '^\[bench\]' "$log" | grep -oE 'loaded in [0-9.]+ s' | head -1)
        echo "[$n $k] ${line#\[bench\] } | $load | thermal $th | free disk $disk MB" >> "$SUM"
        ms=$(print -r -- "$line" | grep -oE '[0-9.]+ ms/frame mean' | cut -d' ' -f1)
        (( ${ms%.*} > 150 )) && { echo "== $ms ms/frame: not on the Neural Engine, stopping" >> "$SUM"; exit 4; }
        break
      fi
    fi
    echo "[$n $k] attempt $attempt gave no result" >> "$SUM"
    grep -hiE "error|invalidated|jetsam|memory pressure" "$log" | head -2 | sed 's/^/   /' >> "$SUM"
    sleep 20
  done
  sleep 10
done
echo "== $(date '+%H:%M:%S') done" >> "$SUM"
