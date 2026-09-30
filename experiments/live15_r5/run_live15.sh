#!/usr/bin/env bash
# Live 15-stream WebRTC test of the r5 engines on a private :8300 server from this worktree (run under box_guard).
#   run_live15.sh <ARM> <RUN dir> <overrides colon-list> <stages: s0,ramp,soak> [ramp levels, default "5 10 15"]
# Isolation: 127.0.0.1:8300 only; private runtime dir; no Lingua control plane, no TURN/STUN, no S3; never touches
# /workspace/logs/musetalk (the user's coturn pid file), the user's launchers or ports 8000/8200. The server is a
# child of this script, so box_guard's RAM watchdog covers it, and the EXIT trap always stops it.
set -uo pipefail
ARM=$1; RUN=$2; OVR=$3; STAGES=$4; LEVELS="${5:-5 10 15}"
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); E=$R/experiments/live15_r5; PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
cd $R; mkdir -p $RUN/runtime $RUN/lt2 $RUN/traces
log() { printf '[live15 %s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $RUN/driver.log; }
ss -ltn | grep -q ':8300 ' && { log "port 8300 busy; refusing"; exit 2; }
touch $RUN/start.stamp
TURN_PID=$(cat /workspace/logs/musetalk/turnserver.pid 2>/dev/null)
{ echo "arm=$ARM overrides=$OVR stages=$STAGES levels=$LEVELS"; echo "git $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)";
  echo "coturn_pid_before=$TURN_PID alive=$(ps -p ${TURN_PID:-0} -o comm= 2>/dev/null)"; for f in ${OVR//:/ }; do echo "--- $f"; cat $f; done; } > $RUN/README.txt
cleanup() {
  bash $E/stop_server.sh $RUN >> $RUN/driver.log 2>&1
  touch $RUN/STOP_SAMPLER
  echo "coturn_pid_after=$TURN_PID alive=$(ps -p ${TURN_PID:-0} -o comm= 2>/dev/null)" >> $RUN/README.txt
  log "stopped; files changed under /workspace/MuseTalk since start: $(find /workspace/MuseTalk -xdev -newer $RUN/start.stamp -not -path '*/__pycache__/*' 2>/dev/null | head -5 | tr '\n' ' ')"
}
trap cleanup EXIT
L="env -u LINGUA_WORKER_TOKEN -u LINGUA_CONTROL_PLANE_BASE_URL -u LINGUA_WORKER_REGISTER_URL -u LINGUA_WORKER_HEARTBEAT_URL \
 REPO_ROOT=$R VENV_PATH=${LIVE15_VENV:-/workspace/.venvs/musetalk_trt_stagewise} HOST=127.0.0.1 PORT=8300 MUSETALK_RECIPE=fast300 \
 MUSETALK_RUNTIME_DIR=$RUN/runtime MUSETALK_ENV_OVERRIDES_FILE=$OVR WEBRTC_STUN_URLS= WEBRTC_RELAY_ENABLED=0 WEBRTC_TURN_AUTOSTART=0 \
 LINGUA_CONTROL_PLANE_ENV_FILE=$RUN/no-control-plane.env"
$L bash scripts/run_musetalk_server.sh --host 127.0.0.1 --port 8300 --print-env > $RUN/print_env.txt 2>&1
$L taskset -c 0-11,16-27 bash scripts/run_musetalk_server.sh --host 127.0.0.1 --port 8300 >> $RUN/api_server_8300.log 2>&1 &
echo $! > $RUN/server.pid
bash $E/sample_box.sh $RUN > $RUN/sampler.log 2>&1 &
python3 $E/cpu_sampler.py $RUN > $RUN/cpu_sampler.log 2>&1 &
log "server pid $(cat $RUN/server.pid); waiting for /health"
t0=$SECONDS
until curl -fsS -m 3 http://127.0.0.1:8300/health 2>/dev/null | grep -q '"ok": *true'; do
  kill -0 $(cat $RUN/server.pid) 2>/dev/null || { log "server exited during startup"; tail -30 $RUN/api_server_8300.log; exit 3; }
  (( SECONDS - t0 < 900 )) || { log "startup timeout"; exit 3; }
  sleep 5
done
log "healthy after $((SECONDS - t0)) s"
$PY -B scripts/musetalk_host_profile.py verify-log --log $RUN/api_server_8300.log --expect-vae taesd_trt --expect-unet trt_stagewise --timeout 30 > $RUN/verify.txt 2>&1
V=$?
grep -E 'HLS GPU scheduler started|300fps flags|WebRTC media flags|UNet backend active|TAESD TRT backend|VAE decode backend|WebRTC H.264 impl|GPU-aware runtime defaults|worker-control' $RUN/api_server_8300.log | cut -c1-400 > $RUN/startup_lines.txt
bad=$(grep -cE 'Traceback|using compiled TAESD|UNet backend: PyTorch|falling back' $RUN/api_server_8300.log)
cp=$(curl -fsS http://127.0.0.1:8300/worker/state | $PY -c 'import json,sys; print(json.load(sys.stdin).get("control_plane_requested"))')
lc=$(curl -fsS 'http://127.0.0.1:8300/webrtc/sessions/stats?view=lifetime' | $PY -c 'import json,sys; print(json.load(sys.stdin)["server"].get("lifetime_counters"))')
log "verify-log rc=$V bad_lines=$bad control_plane_requested=$cp lifetime_counters=$lc"
cat $RUN/startup_lines.txt | tee -a $RUN/driver.log
grep -q 'max_combined_batch_size=16, fixed_batch_sizes=\[16\]' $RUN/startup_lines.txt || { log "INVALID: scheduler shape is not 16/[16]"; exit 4; }
[ $V -eq 0 ] && [ "$bad" = 0 ] && [ "$cp" = False ] && [ "$lc" = True ] || { log "INVALID: backend/isolation verification failed"; exit 4; }
AV=$(cat $E/avatars.txt)
log "pre-warming 15 avatars"
rss0=$(awk '/VmRSS/{print $2}' /proc/$(cat $RUN/server.pid)/status)
for a in ${AV//,/ }; do
  s=$(date +%s.%N)
  st=$(curl -fsS -m 200 -X POST "http://127.0.0.1:8300/avatars/$a/cache/warm?batch_size=16&wait=true&timeout_seconds=180" | $PY -c 'import json,sys; print(json.load(sys.stdin).get("status"))')
  echo "$a status=$st warm_s=$(python3 -c "print(round($(date +%s.%N)-$s,2))") rss_kb=$(awk '/VmRSS/{print $2}' /proc/$(cat $RUN/server.pid)/status) avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)" | tee -a $RUN/warm.txt
done
log "warm done: server RSS $rss0 -> $(awk '/VmRSS/{print $2}' /proc/$(cat $RUN/server.pid)/status) kB; evictions: $(grep -c 'Evicted LRU avatar' $RUN/api_server_8300.log)"
C="--base-url http://127.0.0.1:8300 --avatar-ids $AV --audio-dir experiments/throughput300_candidate/audio_corpus --audio-class turn \
 --ignore-ice-servers --musetalk-fps 20 --playback-fps 20 --batch-size 16 --chunk-duration 1 --ring 64 --turn-gap-s 1.0 \
 --settle-s 10 --cooldown-s 30 --abort-below-gb ${LIVE15_ABORT_GB:-3.5} --post-retries 20 --max-consecutive-post-failures 5 \
 --peers-per-shard ${LIVE15_PEERS_PER_SHARD:-1}"  # 1 = one client process per viewer: a join cannot stall other viewers
foreign() { [ -e $RUN/FOREIGN_GPU ] && { log "FOREIGN GPU app appeared: $(cat $RUN/FOREIGN_GPU); aborting"; exit 5; }; }
report() {  # report <label> <level>
  local dir=$RUN/traces/$1/n$(printf %02d $2)
  $PY $E/live_trace_report.py $dir --json $RUN/$1_n$2_trace.json --md $RUN/$1_n$2_trace.md > /dev/null 2>&1
  $PY -c "import json; s=json.load(open('$RUN/$1_n$2_trace.json'))['summary']; print(json.dumps(s))" | tee -a $RUN/driver.log
}
verdict() { grep -hE '^(PASS|FAIL|INVALID) level' $RUN/$1.out | tail -1 | cut -c1-600 | tee -a $RUN/driver.log; }
if [[ ",$STAGES," == *,s0,* ]]; then
  log "S0 smoke: N=1 then N=3, 2 turns"
  $PY load_test_webrtc_v2.py $C --levels 1,3 --turns 2 --min-steady-s 10 --warmup-s 5 --label s0_$ARM --out-dir $RUN/lt2 \
     --trace-dir $RUN/traces/s0 > $RUN/s0_$ARM.out 2>&1
  grep -hE '^(PASS|FAIL|INVALID) level' $RUN/s0_$ARM.out | cut -c1-500 | tee -a $RUN/driver.log
  report s0 3; foreign
  $PY -c "import json,sys; s=json.load(open('$RUN/s0_n3_trace.json'))['summary']; sys.exit(0 if (s['min_pts_join'] or 0)>=0.99 else 1)" \
    || { log "S0 gate failed (pts join); stopping before the ramp"; exit 6; }
fi
PASSED=0
if [[ ",$STAGES," == *,ramp,* ]]; then
  for N in $LEVELS; do
    log "RAMP N=$N: real staggered joins 5 s apart, 300 s after each join"
    $PY load_test_webrtc_v2.py $C --levels $N --stagger-s 5 --stagger-join --duration-s 300 --turns 60 --min-steady-s 240 \
       --label ramp_$ARM --out-dir $RUN/lt2 --trace-dir $RUN/traces/ramp > $RUN/ramp_${ARM}_n$N.out 2>&1
    verdict ramp_${ARM}_n$N; report ramp $N; foreign
    ok=$($PY -c "import json; print(json.load(open('$RUN/ramp_n${N}_trace.json'))['summary']['all_pass'])")
    inv=$(grep -c '^INVALID level' $RUN/ramp_${ARM}_n$N.out)
    if [ "$ok" = True ] && [ "$inv" = 0 ]; then PASSED=$N; else log "RAMP N=$N did not pass (trace all_pass=$ok invalid=$inv); stopping the ramp"; break; fi
    avail=$(awk -F, 'NR>1 && $2!="" {if(min==""||$2<min)min=$2} END{print int(min/1048576)}' $RUN/box.csv)
    log "MemAvailable minimum so far: ${avail} GB"
  done
fi
if [[ ",$STAGES," == *,soak,* ]]; then
  SN=${LIVE15_SOAK_N:-15}; REC=${LIVE15_SOAK_RECORD-0,$((SN - 1))}  # LIVE15_SOAK_RECORD= (empty) turns observers off
  if [ "$PASSED" != "$SN" ] && [[ ",$STAGES," == *,ramp,* ]]; then log "no soak: the ramp did not pass N=$SN"; exit 0; fi
  log "SOAK: $SN streams, joins 20 s apart, 3320 s of speech per stream, observers on streams '${REC}' at 1800 s"
  $PY load_test_webrtc_v2.py $C --levels $SN --stagger-s 20 --stagger-join --duration-s 3320 --turns 400 --min-steady-s 3000 \
     ${REC:+--record-streams $REC --record-at-s 1800 --record-seconds 90} --label soak_$ARM --out-dir $RUN/lt2 \
     --trace-dir $RUN/traces/soak > $RUN/soak_$ARM.out 2>&1
  verdict soak_$ARM; report soak $SN; foreign
fi
log "driver done"
