#!/usr/bin/env bash
# 1 Hz box sampler for a live run: box.csv (RAM, server RSS/threads/CPU), gpu.csv, gpu_apps.csv (5 s), sched.jsonl (10 s).
# If a GPU compute app that is not our server appears (e.g. the user's :8000 server coming back), it writes
# $RUN/FOREIGN_GPU and stops our server immediately, so the foreign app is never starved by this test.
#   sample_box.sh <RUN>
RUN=$1; PIDF=$RUN/server.pid
echo "t,mem_available_kb,shmem_kb,server_rss_kb,server_hwm_kb,server_threads,server_cpu_ticks,load1" > $RUN/box.csv
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader -l 1 > $RUN/gpu.csv 2>/dev/null &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
i=0
while [ ! -e $RUN/STOP_SAMPLER ]; do
  P=$(cat $PIDF 2>/dev/null)
  ma=$(awk '/MemAvailable/{print $2}' /proc/meminfo); sh=$(awk '/^Shmem:/{print $2}' /proc/meminfo)
  if [ -n "$P" ] && [ -r /proc/$P/status ]; then
    rss=$(awk '/VmRSS/{print $2}' /proc/$P/status); hwm=$(awk '/VmHWM/{print $2}' /proc/$P/status)
    thr=$(awk '/Threads/{print $2}' /proc/$P/status); cpu=$(awk '{print $14+$15}' /proc/$P/stat)
  else rss=; hwm=; thr=; cpu=; fi
  echo "$(date +%s.%N),$ma,$sh,$rss,$hwm,$thr,$cpu,$(cut -d' ' -f1 /proc/loadavg)" >> $RUN/box.csv
  if (( i % 5 == 0 )); then
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
    echo "$(date +%s) ${apps//$'\n'/ | }" >> $RUN/gpu_apps.csv
    if [ -n "$P" ]; then
      for ap in $(echo "$apps" | cut -d, -f1 | tr -d ' '); do
        [ -z "$ap" ] && continue
        if [ "$ap" != "$P" ] && ! grep -qs "^PPid:\s*$P$" /proc/$ap/status; then
          echo "$(date -u +%FT%TZ) foreign GPU pid $ap: stopping our server" | tee $RUN/FOREIGN_GPU
          kill -INT $P 2>/dev/null; sleep 5; kill -KILL $P 2>/dev/null
        fi
      done
    fi
  fi
  if (( i % 10 == 0 )) && [ -n "$P" ]; then
    curl -s -m 3 http://127.0.0.1:${PORT:-8300}/stats 2>/dev/null | python3 -c "import json,sys,time
try:
    d=json.load(sys.stdin); h=d.get('hls_scheduler') or d.get('scheduler') or {}
    print(json.dumps({'t':time.time(),'capacity':h.get('capacity'),'pipeline':h.get('pipeline'),'jobs':len(h.get('jobs') or [])}))
except Exception: pass" >> $RUN/sched.jsonl
  fi
  i=$((i+1)); sleep 1
done
