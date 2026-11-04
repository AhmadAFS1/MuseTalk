#!/bin/zsh
# run_phone_ab.sh <out dir> "<order>" <arm>=<app args> ...
#   run_phone_ab.sh $OUT "A D E E D A" "A=--unet unet_anec0_int8w_b1_chunked --infer-workers 1" \
#     "D=--unet unet_anev2fc0_int8w_b1_chunked --infer-workers 2" "E=--unet unet_anev2fc0_int8w_b1_chunked --infer-workers 3"
# Interleaved runs of the harness app's Lipsync tab (`--lipsync --exit --silent` plus the arm's args, which must
# name the arm's `--unet`) on a connected iPhone with the harness app installed. Order the arms as a mirror (A B B A)
# so the phone warming up hits them equally; arms on different UNets recompile on every switch (§5.3 rule 9).
# - First a check: each arm's UNet for 20 frames on the Neural Engine. It compiles what is not cached, reports free
#   disk and iOS's thermal level, and stops the script if a UNet is not on the Neural Engine (over 150 ms/frame: a
#   phone in that state runs it on the CPU at ~1 s/frame; healthy is 53-72 ms on the A14). Every probe and every run
#   is held to the same test (a run's Neural Engine stage per worker), so a phone that loses the Neural Engine midway
#   stops the script instead of adding CPU runs to the results.
# - No fixed cooldown: the runs start at "nominal" or "fair". At "serious" or "critical" the phone idles 60 s at a time,
#   checked with a short probe, for up to 15 min, then the runs start anyway and the summary says so.
# - Every result carries the thermal level at the start and end of the turn and each change in between
#   (`LipsyncSession`), and the summary prints them.
# - The check, the probes and the runs all run under a watchdog: a SoulX app appearing, or a step past its limit
#   (RUN_TIMEOUT, 150 s; twice that for an arm with --gpu-unet, four times for the check), stops that step, which is
#   retried once the phone is free, twice at most. The SoulX FlashHead tests share the phone.
# Env: UDID (default the iPhone 12 Pro), APP (a built MuseTalkBench.app to install first; a reinstall empties the Core
# ML compile cache, so the check compiles, and frees the space those caches held), WAIT_MIN (how long each wait for the
# phone may take, default 600), RUN_TIMEOUT.
# Writes <out dir>/summary.txt, results.jsonl (each run's LIPSYNC_RESULT plus "run") and one log per attempt.
set -u
(( $# >= 3 )) || { sed -n '2,4p' "$0"; exit 2; }
OUT=$1; ORDER=(${=2}); shift 2
UDID=${UDID:-00008101-00123DA43A51003A}; BUNDLE=com.lingua.musetalkbench
WAIT_MIN=${WAIT_MIN:-600}; RUN_TIMEOUT=${RUN_TIMEOUT:-150}
typeset -A ARGS
for a in "$@"; do ARGS[${a%%=*}]=${a#*=}; done
for k in $ORDER; do (( ${+ARGS[$k]} )) || { echo "arm $k has no args"; exit 2; }; done
unets=()
for k in ${(k)ARGS}; do
  w=(${=ARGS[$k]}); i=${w[(i)--unet]}
  (( i < ${#w} )) || { echo "arm $k names no --unet"; exit 2; }
  unets+=(${w[i+1]})
done
unets=(${(u)unets})
mkdir -p "$OUT"; SUM=$OUT/summary.txt; : > "$SUM"; : > "$OUT/results.jsonl"
BENCH=(--autorun --sustained 0 --chain 0 --no-gpu --no-b4 --no-whisper --no-plan --no-reload --units CPU_AND_NE --exit)
disk=""

process_query() {
  # A failed query is unknown, never evidence that the shared phone is free.
  xcrun devicectl device info processes --device "$UDID" --timeout 15 --json-output "$OUT/processes.json" \
    > "$OUT/process-query.log" 2>&1
}
process_state() {  # free/busy; nonzero for a failed query or an unfamiliar/malformed process list
  process_query || return 2
  python3 -I - "$OUT/processes.json" <<'PY_STATE'
import json, sys
try:
    rows = json.load(open(sys.argv[1]))["result"]["runningProcesses"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("no process inventory")
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("executable"), str) or type(r.get("processIdentifier")) is not int or r["processIdentifier"] <= 0:
            raise ValueError("invalid process record")
    print("busy" if any("soulx" in r["executable"].lower() for r in rows) else "free")
except (OSError, ValueError, TypeError, KeyError):
    sys.exit(2)
PY_STATE
}
musetalk_pids() {  # exact harness executable only; never SoulX or another MuseTalk service
  process_query || return 2
  python3 -I - "$OUT/processes.json" <<'PY_PIDS'
import json, sys
from urllib.parse import unquote, urlparse
try:
    rows = json.load(open(sys.argv[1]))["result"]["runningProcesses"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("no process inventory")
    for r in rows:
        exe, pid = r["executable"], r["processIdentifier"]
        if not isinstance(exe, str) or type(pid) is not int or pid <= 0:
            raise ValueError("invalid process record")
        if unquote(urlparse(exe).path).endswith("/MuseTalkBench.app/MuseTalkBench"):
            print(pid)
except (OSError, ValueError, TypeError, KeyError):
    sys.exit(2)
PY_PIDS
}
stop_musetalk() {
  local pids device_pid
  pids=$(musetalk_pids) || return 1
  for device_pid in ${=pids}; do
    xcrun devicectl device process terminate --device "$UDID" --pid "$device_pid" --kill --timeout 10 \
      >> "$OUT/terminate.log" 2>&1 || true  # a racing normal app exit is harmless; verify below
  done
  pids=$(musetalk_pids) || return 1
  [[ -z $pids ]]
}
ready() {
  xcrun devicectl list devices 2>/dev/null | grep "$UDID" | grep -qv unavailable || return 1
  xcrun devicectl device info lockState --device "$UDID" 2>/dev/null | grep -q "passcodeRequired: false" || return 1
  [[ $(process_state) == free ]]
}
wait_ready() {
  local dl=$(( $(date +%s) + WAIT_MIN * 60 ))
  until ready; do
    (( $(date +%s) > dl )) && { echo "== phone not ready within $WAIT_MIN min" >> "$SUM"; exit 3; }
    sleep 20
  done
}
launch() {  # <log> <app args...>: the app with its console attached; the exit status is devicectl's
  local log=$1; shift
  print -r -- "== $(date '+%H:%M:%S') $*" > "$log"
  xcrun devicectl device process launch --device "$UDID" --console --terminate-existing "$BUNDLE" "$@" >> "$log" 2>&1
}
guarded() {  # <log> <limit s> <app args...>: launch under the watchdog; 1 when it had to stop the app
  local log=$1 limit=$2; shift 2
  launch "$log" "$@" &
  local pid=$! start=$(date +%s) why="" phone_state=""
  while kill -0 $pid 2>/dev/null; do
    sleep 5
    phone_state=$(process_state)
    if [[ $phone_state == busy ]]; then why="SoulX app appeared"
    elif [[ $phone_state != free ]]; then why="phone process inventory unavailable"
    elif (( $(date +%s) - start > limit )); then why="ran past $limit s"; fi
    if [[ -n $why ]]; then
      pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null
      wait $pid 2>/dev/null
      # --console forwards SIGTERM, but explicitly stop and verify only this exact harness executable too.
      if ! stop_musetalk; then
        echo "== cannot verify MuseTalkBench stopped after $why; stopping the script" >> "$SUM"
        exit 7
      fi
      echo "== $(date '+%H:%M:%S') stopped ${log:t}: $why" >> "$SUM"
      return 1
    fi
  done
  wait $pid 2>/dev/null
  return 0
}
failure() {  # <log>: the likely reason a step gave no result, into the summary
  grep -hiE "error|invalidated|terminated due|signal [0-9]|jetsam|memory pressure" "$1" | head -2 | sed "s/^/   /" >> "$SUM"
}
level() {  # <bench log>: the last thermal level the app printed (at launch, then each change)
  grep -h '^\[bench\]' "$1" | grep -oE 'thermal (state -> )?[a-z]+' | tail -1 | awk '{print $NF}'
}
cool() { [[ $1 == nominal || $1 == fair ]]; }
bench_ms() { grep -h '^\[bench\]' "$1" | grep -oE "[0-9.]+ ms/frame mean" | cut -d' ' -f1; }
off_ane() {  # <what> <ms per frame>: stop the script when a UNet frame took more than 150 ms
  (( ${2%.*} > 150 )) || return 0
  echo "== $1 took $2 ms per UNet frame, so not on the Neural Engine (free disk at the check: ${disk:-?} MB; a nearly full disk and a SoulX interruption have both done this): stopping" >> "$SUM"
  exit 4
}
run_ms() {  # <log>: the run's Neural Engine stage per worker
  grep -h '^LIPSYNC_RESULT ' "$1" | head -1 | sed 's/^LIPSYNC_RESULT //' \
    | python3 -I -c 'import sys, json, math; d = json.load(sys.stdin); t = d["stage_infer_ms"]; assert math.isfinite(t) and t > 0; print("%.1f" % (t / max(1, d.get("infer_workers") or 1)))' 2>/dev/null
}
summarize() {  # <log> <n> <arm>; fails when the result does not parse
  grep -h '^LIPSYNC_RESULT ' "$1" | head -1 | sed 's/^LIPSYNC_RESULT //' | python3 -I -c '
import sys, json
import math, shlex
n, k, out, arm_args = sys.argv[1:5]
words = shlex.split(arm_args)
def option(name, default=None):
    found = [words[i + 1] for i, w in enumerate(words[:-1]) if w == name]
    return found[-1] if found else default
d = {"run": "%s_%s" % (n, k), **json.loads(sys.stdin.read())}
if d["unet"] != option("--unet") or d["infer_workers"] != int(option("--infer-workers", "1")):
    raise ValueError("result model/workers do not match the requested arm")
if not (d["frames"] > 0 and d["fps"] > 0 and 0 <= d["generated"] <= d["frames"] and 0 <= d["shown"] <= d["frames"]
        and math.isfinite(d["seconds"]) and d["seconds"] > 0 and math.isfinite(d["generated_fps"])
        and d["generated_fps"] >= 0 and math.isfinite(d["stage_infer_ms"]) and d["stage_infer_ms"] > 0):
    raise ValueError("invalid frame/timing result")
changes = ", ".join("%s at %.1f s" % (c["state"], c["t_s"]) for c in d.get("thermal_changes", []))
gpu = ", GPU %d frames at %.0f ms" % (d["gpu_generated"], d["gpu_ms"]) if d.get("gpu_workers") else ""
line = "[%s %s] %.2f fps (%d of %d generated, %d shown), ANE stage %.1f ms, %s workers%s, skipped %d, late %d, stalls %d, thermal %s -> %s%s" % (
    n, k, d["generated_fps"], d["generated"], d["frames"], d["shown"], d["stage_infer_ms"], d.get("infer_workers"), gpu,
    d["skipped"], d["dropped_late"], d["stalls"], d.get("thermal_start", "?"), d.get("thermal_end", "?"),
    " (%s)" % changes if changes else "")
with open(out, "a") as f:
    f.write(json.dumps(d, sort_keys=True) + "\n")
print(line)
' "$2" "$3" "$OUT/results.jsonl" "${ARGS[$3]}" >> "$SUM" 2>> "$OUT/summarize_errors.txt"
}

echo "== $(date '+%H:%M:%S') order $ORDER; UNets ${(j:, :)unets}" >> "$SUM"
for k in ${(ko)ARGS}; do echo "   $k: ${ARGS[$k]}" >> "$SUM"; done
wait_ready
if [[ -n ${APP:-} ]]; then
  xcrun devicectl device install app --device "$UDID" "$APP" > "$OUT/install.log" 2>&1 \
    || { echo "== install failed (install.log)" >> "$SUM"; exit 5; }
  echo "== $(date '+%H:%M:%S') installed $APP" >> "$SUM"
fi

check=""
for attempt in 1 2 3; do
  wait_ready
  log=$OUT/check_a$attempt.log
  if guarded "$log" $(( RUN_TIMEOUT * 4 )) "${BENCH[@]}" --frames 20 --models "${(j:,:)unets}"; then
    mss=($(bench_ms "$log"))
    (( ${#mss} == ${#unets} )) && { check=$log; break; }
    echo "== the check gave ${#mss} of ${#unets} UNet results" >> "$SUM"
  fi
  failure "$log"; sleep 30
done
[[ -n $check ]] || { echo "== no check after 3 attempts" >> "$SUM"; exit 4; }
grep -h '^\[bench\]' "$check" | grep -E "free disk|device:|ms/frame mean|thermal state" | head -8 | sed 's/^/[check] /' >> "$SUM"
disk=$(grep -h '^\[bench\] free disk' "$check" | head -1 | grep -oE '[0-9]+' | head -1)
for ms in $mss; do off_ane "the check" $ms; done
[[ -n $disk ]] && (( disk < 4000 )) && echo "== free disk ${disk:-?} MB: every UNet compile writes about 1.2 GB, and at 2.5 GB free the Neural Engine compile has failed" >> "$SUM"

lvl=$(level "$check"); gate_start=$(date +%s)
until cool "$lvl"; do
  if (( $(date +%s) - gate_start >= 900 )); then
    echo "== still ${lvl:-unknown} after $(( ($(date +%s) - gate_start) / 60 )) min: running anyway" >> "$SUM"; break
  fi
  echo "== $(date '+%H:%M:%S') thermal ${lvl:-unknown}: idling 60 s" >> "$SUM"
  sleep 60; wait_ready
  log=$OUT/probe_$(date +%H%M%S).log; lvl=""
  if guarded "$log" $RUN_TIMEOUT "${BENCH[@]}" --frames 5 --accuracy-frames 8 --models "${unets[1]}"; then
    ms=($(bench_ms "$log")); (( ${#ms} )) && off_ane "the probe" ${ms[1]}
    lvl=$(level "$log")
  fi
done
echo "== $(date '+%H:%M:%S') thermal ${lvl:-unknown}: starting the runs" >> "$SUM"

n=0
for k in $ORDER; do
  n=$((n + 1))
  limit=$RUN_TIMEOUT; [[ ${ARGS[$k]} == *--gpu-unet* ]] && limit=$(( RUN_TIMEOUT * 2 ))
  success=0
  for attempt in 1 2 3; do
    wait_ready
    log=$OUT/${n}_${k}_a$attempt.log
    if guarded "$log" $limit --lipsync --exit --silent ${=ARGS[$k]} && grep -q '^LIPSYNC_RESULT ' "$log"; then
      ms=$(run_ms "$log"); [[ -n $ms ]] && off_ane "run $n $k" $ms
      if summarize "$log" $n $k; then success=1; break; fi
      echo "[$n $k] attempt $attempt: the result did not parse (summarize_errors.txt)" >> "$SUM"
    else
      echo "[$n $k] attempt $attempt gave no result at $(date '+%H:%M:%S')" >> "$SUM"
    fi
    failure "$log"; sleep 30
  done
  (( success )) || { echo "== run $n $k failed after 3 attempts; stopping" >> "$SUM"; exit 6; }
  sleep 10
done
python3 -I - "$OUT/results.jsonl" "${ORDER[@]}" <<'PY_COMPLETE' >> "$SUM" 2>&1 || exit 6
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
expected = [f"{i}_{arm}" for i, arm in enumerate(sys.argv[2:], 1)]
if [r["run"] for r in rows] != expected:
    sys.exit("== incomplete run count/order")
if len({(r["avatar"], r["fps"], r["frames"]) for r in rows}) != 1:
    sys.exit("== arms did not use the same avatar/fps/frame count")
print(f"== verified {len(rows)} ordered results with matched inputs")
PY_COMPLETE
echo "== $(date '+%H:%M:%S') done" >> "$SUM"
