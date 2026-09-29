#!/usr/bin/env bash
# Stop the live-test server of <RUN> cleanly (only if that pid really is our :8300 server from this worktree).
#   stop_server.sh <RUN>
RUN=$1; PID=$(cat $RUN/server.pid 2>/dev/null) || exit 0
[ -n "$PID" ] && [ -d /proc/$PID ] || exit 0
[[ "$(readlink /proc/$PID/cwd)" == /workspace/MuseTalk-perf300 ]] && tr '\0' ' ' </proc/$PID/cmdline | grep -q -- '--port 8300' || { echo "pid $PID is not our :8300 server; not touching it"; exit 0; }
for s in $(curl -s -m 5 http://127.0.0.1:8300/webrtc/sessions/stats | python3 -c "import json,sys
try: [print(x['session_id']) for x in json.load(sys.stdin).get('sessions',[])]
except Exception: pass"); do curl -s -m 5 -X DELETE http://127.0.0.1:8300/webrtc/sessions/$s >/dev/null; done
kill -INT $PID 2>/dev/null
for i in $(seq 45); do kill -0 $PID 2>/dev/null || break; sleep 1; done
kill -INT $PID 2>/dev/null; sleep 5; kill -KILL $PID 2>/dev/null
# a profiler shim (py-spy record -- python api_server.py) may leave its child running: stop our :8300 server by name,
# only for processes whose cwd is this worktree
for p in $(pgrep -f "api_server.py --host 127.0.0.1 --port 8300"); do
  [[ "$(readlink /proc/$p/cwd)" == /workspace/MuseTalk-perf300 ]] || continue
  kill -INT $p 2>/dev/null; sleep 3; kill -INT $p 2>/dev/null; sleep 5; kill -KILL $p 2>/dev/null
done
echo "server $PID stopped"
