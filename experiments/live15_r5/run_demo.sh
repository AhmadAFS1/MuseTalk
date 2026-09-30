#!/usr/bin/env bash
# Watchable live demo: the r5 server with the live-test fixes on 0.0.0.0:6006 (public port from VAST_TCP_PORT_6006),
# media through the machine's running coturn, one group of N sessions, and a driver that keeps every session talking
# while a person watches the group wall in a browser. Run under box_guard.
#   run_demo.sh <RUN dir> [N=15] [minutes=45] [avatar]
# Never starts or stops coturn, never binds 8000/8200, never registers with the Lingua control plane.
set -uo pipefail
RUN=$1; N=${2:-15}; MIN=${3:-45}; AVATAR=${4:-chinese_bob_pink_bedroom_talking_3373c10448}
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); E=$R/experiments/live15_r5; PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
export PORT=6006
cd $R; mkdir -p $RUN/runtime
log() { printf '[demo %s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $RUN/driver.log; }
ss -ltn | grep -q ":$PORT " && { log "port $PORT busy; refusing"; exit 2; }
PUB_IP=$(tr '\0' '\n' </proc/1/environ | sed -n 's/^PUBLIC_IPADDR=//p'); PUB_PORT=$(tr '\0' '\n' </proc/1/environ | sed -n "s/^VAST_TCP_PORT_$PORT=//p")
[ -n "$PUB_IP" ] && [ -n "$PUB_PORT" ] || { log "no public mapping for port $PORT"; exit 2; }
TURN_PID=$(pgrep -x turnserver | head -1); [ -n "$TURN_PID" ] || { log "coturn is not running; refusing (this script never starts it)"; exit 2; }
echo "demo N=$N minutes=$MIN avatar=$AVATAR git $(git rev-parse HEAD) coturn_pid=$TURN_PID" > $RUN/README.txt
cleanup() {
  touch $RUN/STOP_DEMO $RUN/STOP_SAMPLER
  bash $E/stop_server.sh $RUN >> $RUN/driver.log 2>&1
  log "stopped; coturn pid $TURN_PID alive=$(ps -p $TURN_PID -o comm= 2>/dev/null)"
}
trap cleanup EXIT
OVR="$E/demo.env:$E/loopfix.env:$E/serve.env:$E/common.env"
# The TURN env file may name a TCP fallback listener (TURN_TCP_FALLBACK_LISTEN_PORT) that the running coturn does not
# open. Use only what coturn actually listens on: the server reaches it on loopback, browsers through the public
# UDP mapping of the same port.
TURN_PORT=$(sed -n 's/^listening-port=//p' /tmp/musetalk-turnserver-tcp-relay.conf 2>/dev/null | head -1); TURN_PORT=${TURN_PORT:-3478}
TURN_UDP_PUB=$(tr '\0' '\n' </proc/1/environ | sed -n "s/^VAST_UDP_PORT_$TURN_PORT=//p")
[ -n "$TURN_UDP_PUB" ] || { log "no public UDP mapping for TURN port $TURN_PORT"; exit 2; }
export WEBRTC_TURN_URLS="turn:$PUB_IP:$TURN_UDP_PUB?transport=udp"
TURN_FILE=/workspace/MuseTalk/.env.webrtc-turn.local
if [ "${DEMO_SERVER_RELAY:-0}" = 1 ]; then
  # Both peers relay through coturn: ~3-4 relay ports per call. The running coturn has 41 (49160-49200), so only
  # ~11 calls connect; the rest stay in ICE state "new" (coturn logs "no available ports").
  export WEBRTC_SERVER_TURN_URLS="turn:127.0.0.1:$TURN_PORT?transport=udp"
  LAUNCH="bash scripts/run_webrtc_relay_api_server.sh"
else
  # Default: only browsers relay. coturn runs in this container, so a browser's relay reaches the server's own host
  # candidate directly (the server learns it as peer-reflexive): ~1-2 relay ports per call instead of ~3-4.
  export WEBRTC_ICE_TRANSPORT_POLICY=all WEBRTC_STUN_URLS= WEBRTC_SERVER_TURN_URLS=
  WEBRTC_TURN_USER=$(bash -c 'set -a; source "$1" >/dev/null 2>&1; printf %s "${TURN_USER:-webrtc}"' _ $TURN_FILE)
  WEBRTC_TURN_PASS=$(bash -c 'set -a; source "$1" >/dev/null 2>&1; printf %s "${TURN_PASS:-}"' _ $TURN_FILE)
  export WEBRTC_TURN_USER WEBRTC_TURN_PASS
  [ -n "$WEBRTC_TURN_PASS" ] || { log "no TURN password in $TURN_FILE"; exit 2; }
  LAUNCH="bash scripts/run_musetalk_server.sh"
fi
log "server relay=${DEMO_SERVER_RELAY:-0} browser TURN=$WEBRTC_TURN_URLS server TURN='${WEBRTC_SERVER_TURN_URLS}' policy=${WEBRTC_ICE_TRANSPORT_POLICY:-from TURN file}"
env -u LINGUA_WORKER_TOKEN -u LINGUA_CONTROL_PLANE_BASE_URL -u LINGUA_WORKER_REGISTER_URL -u LINGUA_WORKER_HEARTBEAT_URL \
  REPO_ROOT=$R VENV_PATH=/workspace/.venvs/musetalk_trt_stagewise MUSETALK_RECIPE=fast300 \
  MUSETALK_RUNTIME_DIR=$RUN/runtime MUSETALK_ENV_OVERRIDES_FILE=$OVR LINGUA_CONTROL_PLANE_ENV_FILE=$RUN/no-control-plane.env \
  TURN_ENV_FILE=$TURN_FILE \
  taskset -c 0-11,16-27 $LAUNCH --host 0.0.0.0 --port $PORT >> $RUN/api_server.log 2>&1 &
echo $! > $RUN/server.pid
bash $E/sample_box.sh $RUN > $RUN/sampler.log 2>&1 &
t0=$SECONDS
until curl -fsS -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok": *true'; do
  kill -0 $(cat $RUN/server.pid) 2>/dev/null || { log "server exited during startup"; grep -vE "PASS|credential" $RUN/api_server.log | tail -20; exit 3; }
  (( SECONDS - t0 < 900 )) || { log "startup timeout"; exit 3; }
  sleep 5
done
# the relay launcher execs the server; server.pid is the api_server process after exec
log "healthy after $((SECONDS - t0)) s"
$PY -B scripts/musetalk_host_profile.py verify-log --log $RUN/api_server.log --expect-vae taesd_trt --expect-unet trt_stagewise --timeout 30 > $RUN/verify.txt 2>&1 \
  || { log "INVALID: backend verification failed"; cat $RUN/verify.txt; exit 4; }
grep -E "WEBRTC_ICE_TRANSPORT_POLICY=|WEBRTC_TURN_URLS=|WEBRTC_SERVER_TURN_URLS=|HLS GPU scheduler started|WebRTC media flags" $RUN/api_server.log | cut -c1-300 | tee -a $RUN/driver.log
st=$(curl -fsS -m 200 -X POST "http://127.0.0.1:$PORT/avatars/$AVATAR/cache/warm?batch_size=16&wait=true&timeout_seconds=180" | $PY -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status"), d.get("idle_frame_cache"))')
log "warm $AVATAR: $st"
curl -fsS -m 60 -X POST "http://127.0.0.1:$PORT/webrtc/groups/create?avatar_id=$AVATAR&count=$N&batch_size=16&musetalk_fps=20&playback_fps=20&chunk_duration=1" > $RUN/group_create.json \
  || { log "group create failed"; exit 5; }
GID=$($PY -c "import json; print(json.load(open('$RUN/group_create.json'))['group_id'])")
$PY -c "import json; d=json.load(open('$RUN/group_create.json')); json.dump({'group_id': d['group_id'], 'sessions': [s['session_id'] for s in d['sessions']]}, open('$RUN/group.json','w'))"
rm -f $RUN/group_create.json   # holds the ICE credentials handed to browsers
WALL="http://$PUB_IP:$PUB_PORT/webrtc/groups/$GID/wall"
echo "$WALL" > $RUN/WALL_URL
log "WALL $WALL"
$PY $E/demo_driver.py http://127.0.0.1:$PORT $GID $RUN --minutes $MIN >> $RUN/demo_driver.log 2>&1 &
DRV=$!
end=$((SECONDS + MIN * 60 + 60))
while (( SECONDS < end )) && [ ! -e $RUN/STOP_DEMO ] && [ ! -e $RUN/FOREIGN_GPU ] && kill -0 $(cat $RUN/server.pid) 2>/dev/null; do sleep 10; done
[ -e $RUN/FOREIGN_GPU ] && log "another GPU app appeared: $(cat $RUN/FOREIGN_GPU)"
touch $RUN/STOP_DEMO; wait $DRV 2>/dev/null
log "demo done"
