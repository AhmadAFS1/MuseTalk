#!/usr/bin/env bash
# box_guard.sh - shared-box guard and GPU lease for MuseTalk work on this host.
#
# Plan item 0.1 (docs/musetalk_4070s_300fps_plan_2026-09-27.md). Stdlib tools only
# (bash, awk, flock, setsid, ps, df, du, nvidia-smi).
#
#   box_guard.sh check [options]            print box state; exit 0 only if the box is quiet
#   box_guard.sh run [options] -- cmd ...   run cmd under the GPU lease with RAM protection
#   box_guard.sh lease                      print who holds the GPU lease (exit 1 if held)
#
# Why: a previous run OOM-killed the user's server, and several agent sessions
# (MuseTalk, SoulX, Codex) share one 12 GB GPU and 30 GB of RAM. Every command that
# initialises CUDA, loads a model, builds an engine or benchmarks goes through `run`.
#
# `run` does, in order:
#   1. flock on the lease file (default /workspace/.gpu_lease), waiting up to --wait-min;
#   2. waits (polling every --poll-s) until no GPU compute app is visible and GPU memory
#      used is <= --max-gpu-mem-mib, then requires --settle-s of quiet GPU;
#   3. waits until MemAvailable >= --min-avail-gb and fails fast if disk free < --need-disk-gb;
#   4. snapshots cgroup memory.events oom_kill, nvidia-smi, /dev/shm and disk;
#   5. starts the command in its own process group with oom_score_adj=1000;
#   6. a 1 Hz watchdog kills that process group if MemAvailable < --kill-below-gb
#      (it also records the minimum MemAvailable and the peak RSS of the group);
#   7. returns the command's exit code, or 86 if the watchdog fired, or 87 if the
#      cgroup oom_kill counter rose during a run that otherwise exited 0.
#
# Exit codes of `run` itself: 2 usage, 75 lease/GPU wait timed out, 76 RAM below
# threshold after waiting, 77 disk below threshold, 86 watchdog kill, 87 oom_kill rose.
#
# MemAvailable used everywhere is min(/proc/meminfo MemAvailable, cgroup estimate),
# where the cgroup estimate is memory.max - memory.current + active_file +
# inactive_file + slab_reclaimable (shmem is not counted as reclaimable).
#
# Environment overrides (tests / other hosts):
#   BOX_GUARD_LEASE_FILE (default /workspace/.gpu_lease)
#   BOX_GUARD_LOG        (default /workspace/.gpu_lease.log, one line per run)
#   BOX_GUARD_DISK_PATH  (default /workspace)
#   BOX_GUARD_MEMORY_EVENTS (default /sys/fs/cgroup/memory.events)

set -uo pipefail

LEASE_FILE=${BOX_GUARD_LEASE_FILE:-/workspace/.gpu_lease}
HOLDER_FILE="${LEASE_FILE}.holder"
# Operator pause: while this file exists, waiters stand aside WITHOUT holding the lease and the pause
# time does not count toward --wait-min. Callers that must run during a pause set BOX_GUARD_IGNORE_PAUSE=1.
PAUSE_FILE=${BOX_GUARD_PAUSE_FILE:-${LEASE_FILE}.pause}
LOG_FILE=${BOX_GUARD_LOG:-/workspace/.gpu_lease.log}
DISK_PATH=${BOX_GUARD_DISK_PATH:-/workspace}
CGROUP_DIR=/sys/fs/cgroup
MEMORY_EVENTS=${BOX_GUARD_MEMORY_EVENTS:-$CGROUP_DIR/memory.events}
COTENANT_RE='api_server\.py|run_wall_api|run_local_api|dev_server|drive_wall|trtexec'

log() { printf '[box_guard %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { local rc=$1; shift; log "ERROR: $*"; exit "$rc"; }

# ---------------------------------------------------------------- measurements
meminfo_avail_kb() { awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo; }

cgroup_avail_kb() {
    local max cur
    max=$(cat "$CGROUP_DIR/memory.max" 2>/dev/null || echo max)
    cur=$(cat "$CGROUP_DIR/memory.current" 2>/dev/null || echo "")
    if [[ "$max" == "max" || -z "$cur" ]]; then echo ""; return; fi
    awk -v max="$max" -v cur="$cur" '
        $1=="active_file"||$1=="inactive_file"||$1=="slab_reclaimable" {r+=$2}
        END {printf "%d\n", (max-cur+r)/1024}' "$CGROUP_DIR/memory.stat"
}

avail_kb() {
    local a b
    a=$(meminfo_avail_kb); b=$(cgroup_avail_kb)
    if [[ -n "$b" && "$b" -lt "$a" ]]; then echo "$b"; else echo "$a"; fi
}

kb_to_gb() { awk -v k="$1" 'BEGIN {printf "%.2f", k/1048576}'; }
gb_to_kb() { awk -v g="$1" 'BEGIN {printf "%d", g*1048576}'; }
disk_free_kb() { df -Pk "$DISK_PATH" | awk 'NR==2 {print $4}'; }
oom_kill_count() { awk '$1=="oom_kill" {print $2; f=1} END {if (!f) print 0}' "$MEMORY_EVENTS" 2>/dev/null || echo 0; }
load1() { awk '{print $1}' /proc/loadavg; }
shm_used() { df -Ph /dev/shm | awk 'NR==2 {print $3 " used of " $2}'; }

# "pid, name, MiB" lines; empty when no compute app is visible. Returns 1 if nvidia-smi fails.
gpu_apps() {
    local out
    out=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1) || { echo "$out"; return 1; }
    printf '%s\n' "$out" | grep -v -i -e '^$' -e 'no running' || true
}

# "util mem_used mem_total sm_clock power temp" (space separated)
gpu_stats() {
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,clocks.sm,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' | tr ',' ' '
}

# Own process and all ancestors (so a check run from a shell whose command line
# mentions api_server.py does not report itself).
ancestors() {
    local p=$$
    while [[ -n "$p" && "$p" != "0" ]]; do
        echo "$p"
        p=$(awk '/^PPid:/ {print $2}' "/proc/$p/status" 2>/dev/null)
    done
}

cotenants() {
    local skip
    skip=" $(ancestors | tr '\n' ' ') "
    ps -eo pid=,etimes=,rss=,args= | while read -r pid et rss args; do
        [[ "$skip" == *" $pid "* ]] && continue
        [[ "$args" == *box_guard* ]] && continue
        if [[ "$args" =~ $COTENANT_RE ]]; then
            printf '  pid=%s age=%ss rss=%sMiB %s\n' "$pid" "$et" "$((rss / 1024))" "${args:0:160}"
        fi
    done
}

describe_gpu_apps() {
    local apps="$1" pid name mem cmd
    while IFS=, read -r pid name mem; do
        pid=${pid// /}; [[ -z "$pid" ]] && continue
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)
        printf '  pid=%s mem=%sMiB name=%s cmd=%s\n' "$pid" "${mem// /}" "${name# }" "${cmd:-<not visible in this namespace>}"
    done <<< "$apps"
}

# Processes that have the lease file open (the flock owner is among them).
lease_fd_holders() {
    local f pid target out=""
    target=$(readlink -f "$LEASE_FILE")
    for f in /proc/[0-9]*/fd/*; do
        [[ "$(readlink "$f" 2>/dev/null)" == "$target" ]] || continue
        pid=${f#/proc/}; pid=${pid%%/*}
        [[ " $out " == *" $pid "* ]] || out="$out $pid"
    done
    for pid in $out; do
        printf 'pid=%s cmd=%s; ' "$pid" "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-100)"
    done
}

holder_info() {
    local h; h=$(tr '\n' ' ' < "$HOLDER_FILE" 2>/dev/null)
    if [[ -z "${h// /}" ]]; then h="(no holder record) open by: $(lease_fd_holders)"; fi
    echo "$h"
}

lease_status() {
    # Prints the holder and returns 1 if the lease is currently held by someone else.
    [[ -e "$LEASE_FILE" ]] || { echo "free (lease file absent)"; return 0; }
    if flock -n "$LEASE_FILE" true 2>/dev/null; then
        echo "free"; return 0
    fi
    echo "HELD: $(holder_info)"
    return 1
}

snapshot() {
    # One-line state summary used before/after runs.
    local g; g=$(gpu_stats)
    printf 'avail=%sGB oom_kill=%s disk_free=%sGB shm=%s load1=%s gpu[util%%,memMiB,totMiB,smMHz,W,C]=%s' \
        "$(kb_to_gb "$(avail_kb)")" "$(oom_kill_count)" "$(kb_to_gb "$(disk_free_kb)")" \
        "$(shm_used | tr ' ' '_')" "$(load1)" "${g// /,}"
}

# ---------------------------------------------------------------------- check
cmd_check() {
    local min_avail=6 need_disk=1 max_load=2 max_gmem=600 max_util=5 quiet_s=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --min-avail-gb) min_avail=$2; shift 2 ;;
            --need-disk-gb) need_disk=$2; shift 2 ;;
            --max-load) max_load=$2; shift 2 ;;
            --max-gpu-mem-mib) max_gmem=$2; shift 2 ;;
            --max-gpu-util) max_util=$2; shift 2 ;;
            --quiet-s) quiet_s=$2; shift 2 ;;
            -h|--help) usage; return 0 ;;
            *) die 2 "check: unknown option $1" ;;
        esac
    done

    local reasons=() apps gs util gmem gtot sm pw temp av dk ld oom co lease
    local t_end=$(( $(date +%s) + quiet_s ))
    while :; do
        reasons=()
        if ! apps=$(gpu_apps); then reasons+=("nvidia-smi failed: $apps"); apps=""; fi
        gs=$(gpu_stats); read -r util gmem gtot sm pw temp <<< "$gs"
        [[ -n "$apps" ]] && reasons+=("foreign GPU compute apps present")
        [[ -n "${util:-}" && "${util%.*}" -gt "$max_util" ]] && reasons+=("GPU util ${util}% > ${max_util}%")
        [[ -n "${gmem:-}" && "${gmem%.*}" -gt "$max_gmem" ]] && reasons+=("GPU memory used ${gmem} MiB > ${max_gmem} MiB")
        av=$(avail_kb); [[ "$av" -lt "$(gb_to_kb "$min_avail")" ]] && reasons+=("MemAvailable $(kb_to_gb "$av") GB < ${min_avail} GB")
        dk=$(disk_free_kb); [[ "$dk" -lt "$(gb_to_kb "$need_disk")" ]] && reasons+=("disk free $(kb_to_gb "$dk") GB < ${need_disk} GB")
        ld=$(load1); awk -v l="$ld" -v m="$max_load" 'BEGIN {exit !(l > m)}' && reasons+=("load1 ${ld} > ${max_load}")
        co=$(cotenants); [[ -n "$co" ]] && reasons+=("co-tenant processes running")
        lease=$(lease_status) || reasons+=("GPU lease held")
        [[ ${#reasons[@]} -gt 0 || $(date +%s) -ge $t_end ]] && break
        sleep 5
    done

    echo "== box_guard check $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "GPU compute apps:"; if [[ -n "$apps" ]]; then describe_gpu_apps "$apps"; else echo "  none"; fi
    echo "GPU: util=${util:-?}% mem=${gmem:-?}/${gtot:-?} MiB sm=${sm:-?} MHz power=${pw:-?} W temp=${temp:-?} C"
    echo "MemAvailable: $(kb_to_gb "$av") GB (meminfo $(kb_to_gb "$(meminfo_avail_kb)") GB, cgroup est $(kb_to_gb "$(cgroup_avail_kb)") GB)"
    echo "/dev/shm: $(shm_used)"
    du -sk /dev/shm/* 2>/dev/null | sort -rn | head -5 | awk '{printf "  %7.2f GB %s\n", $1/1048576, $2}'
    echo "Disk free at $DISK_PATH: $(kb_to_gb "$dk") GB"
    echo "Load average: $(cut -d' ' -f1-3 /proc/loadavg) (nproc $(nproc))"
    echo "cgroup oom_kill: $(oom_kill_count)"
    echo "GPU lease ($LEASE_FILE): $lease"
    echo "Co-tenant processes:"; if [[ -n "$co" ]]; then echo "$co"; else echo "  none"; fi
    if [[ ${#reasons[@]} -eq 0 ]]; then
        echo "VERDICT: QUIET"; return 0
    fi
    echo "VERDICT: NOT QUIET"; printf '  - %s\n' "${reasons[@]}"
    return 1
}

# ------------------------------------------------------------------------ run
CHILD_PID=""
CHILD_PGID=""
WATCHDOG_PID=""

kill_child_group() {
    local sig=$1
    [[ -z "$CHILD_PGID" ]] && return
    kill "-$sig" -- "-$CHILD_PGID" 2>/dev/null || kill "-$sig" "$CHILD_PID" 2>/dev/null || true
}

terminate_child() {
    [[ -z "$CHILD_PID" ]] && return
    kill -0 "$CHILD_PID" 2>/dev/null || return
    kill_child_group TERM
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$CHILD_PID" 2>/dev/null || return
        sleep 0.5
    done
    kill_child_group KILL
}

on_signal() {
    log "received $1; terminating child process group ${CHILD_PGID:-none}"
    SIGNALLED=$1
    terminate_child
}

watchdog() {
    # Runs in a background subshell. Kills the child's process group when
    # MemAvailable drops below the threshold. Records the minimum seen.
    local threshold_kb=$1 marker=$2 minfile=$3 rssfile=$4 a min=999999999 rss peak=0
    trap - INT TERM HUP
    while kill -0 "$CHILD_PID" 2>/dev/null; do
        a=$(avail_kb)
        if [[ "$a" -lt "$min" ]]; then min=$a; echo "$min" > "$minfile"; fi
        rss=$(ps -eo pgid=,rss= | awk -v g="$CHILD_PGID" '$1==g {s+=$2} END {print s+0}')
        if [[ "$rss" -gt "$peak" ]]; then peak=$rss; echo "$peak" > "$rssfile"; fi
        if [[ "$a" -lt "$threshold_kb" ]]; then
            echo "MemAvailable $(kb_to_gb "$a") GB < $(kb_to_gb "$threshold_kb") GB at $(date +%H:%M:%S)" > "$marker"
            log "WATCHDOG: MemAvailable $(kb_to_gb "$a") GB < $(kb_to_gb "$threshold_kb") GB; killing process group $CHILD_PGID"
            terminate_child
            return
        fi
        sleep 1
    done
}

cmd_run() {
    local min_avail=6 need_disk=1 wait_min=30 poll_s=30 kill_below=3 settle_s=10 max_gmem=600 max_util=5 label=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --min-avail-gb) min_avail=$2; shift 2 ;;
            --need-disk-gb) need_disk=$2; shift 2 ;;
            --wait-min) wait_min=$2; shift 2 ;;
            --poll-s) poll_s=$2; shift 2 ;;
            --kill-below-gb) kill_below=$2; shift 2 ;;
            --settle-s) settle_s=$2; shift 2 ;;
            --max-gpu-mem-mib) max_gmem=$2; shift 2 ;;
            --max-gpu-util) max_util=$2; shift 2 ;;
            --label) label=$2; shift 2 ;;
            --) shift; break ;;
            -h|--help) usage; return 0 ;;
            *) die 2 "run: unknown option $1 (did you forget '--' before the command?)" ;;
        esac
    done
    [[ $# -gt 0 ]] || die 2 "run: no command given (usage: box_guard.sh run [options] -- cmd ...)"
    local wait_s; wait_s=$(awk -v m="$wait_min" 'BEGIN {printf "%d", m*60}')
    local cmd_str="$*"; label=${label:-${cmd_str:0:80}}

    # 1+2. GPU lease, held ONLY once the GPU is quiet. If a foreign GPU app is present (or an operator
    # pause is active) the lease is released while waiting, so another lease user (e.g. the session that
    # owns that foreign app and wants to stop it) can take the lease in between.
    local lfd started_wait=$(date +%s)
    exec {lfd}<>"$LEASE_FILE" || die 2 "cannot open lease file $LEASE_FILE"
    local deadline=$(( started_wait + wait_s ))
    local apps gs util gmem rest quiet_since busy_since holding=0
    while :; do
        # Operator pause (not counted toward the deadline).
        while [[ -e "$PAUSE_FILE" && "${BOX_GUARD_IGNORE_PAUSE:-0}" != "1" ]]; do
            log "paused by $PAUSE_FILE ($(head -c 200 "$PAUSE_FILE" 2>/dev/null | tr '\n' ' ')); waiting ${poll_s}s without holding the lease"
            sleep "$poll_s"; deadline=$(( deadline + poll_s ))
        done
        while ! flock -w "$(( poll_s < 1 ? 1 : poll_s ))" "$lfd"; do
            if [[ $(date +%s) -ge $deadline ]]; then
                die 75 "timed out after ${wait_min} min waiting for the GPU lease; holder: $(holder_info)"
            fi
            log "waiting for GPU lease; holder: $(holder_info)"
        done
        if [[ -e "$PAUSE_FILE" && "${BOX_GUARD_IGNORE_PAUSE:-0}" != "1" ]]; then
            flock -u "$lfd"; continue
        fi
        holding=1
        printf 'pid=%s since=%s label=%s state=settling\n' "$$" "$(date '+%Y-%m-%dT%H:%M:%S')" "$label" > "$HOLDER_FILE"
        quiet_since=""; busy_since=""
        local release=0
        while :; do
            if ! apps=$(gpu_apps); then die 75 "nvidia-smi failed: $apps"; fi
            gs=$(gpu_stats); read -r util gmem rest <<< "$gs"
            if [[ -z "$apps" && "${gmem%.*}" -le "$max_gmem" && "${util%.*}" -le "$max_util" ]]; then
                busy_since=""
                [[ -z "$quiet_since" ]] && quiet_since=$(date +%s)
                [[ $(( $(date +%s) - quiet_since )) -ge $settle_s ]] && break
                sleep 2; continue
            fi
            quiet_since=""
            if [[ -n "$apps" ]]; then release=1; break; fi
            # No visible app but memory/util busy (another container, or a process just exiting).
            [[ -z "$busy_since" ]] && busy_since=$(date +%s)
            if [[ $(( $(date +%s) - busy_since )) -ge 60 ]]; then release=1; break; fi
            sleep 2
        done
        [[ $release -eq 0 ]] && break
        # Foreign GPU use: give the lease back while waiting.
        : > "$HOLDER_FILE" 2>/dev/null; flock -u "$lfd"; holding=0
        if [[ $(date +%s) -ge $deadline ]]; then
            log "foreign GPU use still present after ${wait_min} min:"; describe_gpu_apps "$apps" >&2
            die 75 "GPU not quiet (util=${util}% mem=${gmem}MiB); giving up"
        fi
        if [[ -n "$apps" ]]; then
            log "foreign GPU compute apps present; released the lease, retrying in ${poll_s}s:"; describe_gpu_apps "$apps" >&2
        else
            log "GPU busy without a visible app (util=${util}% mem=${gmem}MiB) for 60 s; released the lease, retrying in ${poll_s}s"
        fi
        sleep "$poll_s"
    done
    local lease_wait=$(( $(date +%s) - started_wait ))
    printf 'pid=%s since=%s label=%s\n' "$$" "$(date '+%Y-%m-%dT%H:%M:%S')" "$label" > "$HOLDER_FILE"
    trap ': > "$HOLDER_FILE" 2>/dev/null' EXIT
    log "GPU lease acquired after ${lease_wait}s with a quiet GPU (label: $label)"

    # 3. RAM and disk thresholds (fresh --wait-min window, as before).
    deadline=$(( $(date +%s) + wait_s ))
    local need_kb; need_kb=$(gb_to_kb "$min_avail")
    while [[ "$(avail_kb)" -lt "$need_kb" ]]; do
        if [[ $(date +%s) -ge $deadline ]]; then
            die 76 "MemAvailable $(kb_to_gb "$(avail_kb)") GB < ${min_avail} GB after waiting"
        fi
        log "MemAvailable $(kb_to_gb "$(avail_kb)") GB < ${min_avail} GB; waiting ${poll_s}s"
        sleep "$poll_s"
    done
    [[ "$(disk_free_kb)" -ge "$(gb_to_kb "$need_disk")" ]] || \
        die 77 "disk free $(kb_to_gb "$(disk_free_kb)") GB at $DISK_PATH < ${need_disk} GB"

    # 4. Snapshot before.
    local co; co=$(cotenants)
    [[ -n "$co" ]] && { log "note: co-tenant processes are running (not using the GPU):"; echo "$co" >&2; }
    local oom_before; oom_before=$(oom_kill_count)
    local snap_before; snap_before=$(snapshot)
    log "before: $snap_before"
    log "running: $cmd_str"

    # 5. Child in its own process group with oom_score_adj=1000.
    local tmpd; tmpd=$(mktemp -d "${TMPDIR:-/tmp}/box_guard.XXXXXX")
    local marker="$tmpd/watchdog_fired" minfile="$tmpd/min_avail_kb" rssfile="$tmpd/peak_rss_kb"
    SIGNALLED=""
    trap 'on_signal INT' INT
    trap 'on_signal TERM' TERM
    trap 'on_signal HUP' HUP
    local t0; t0=$(date +%s.%N)
    # The child does not inherit the lease descriptor: a daemon it leaks must not hold
    # the lease forever. Such a process is caught by the next run's GPU-app wait instead.
    setsid bash -c 'exec '"$lfd"'>&-; echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true; exec "$@"' \
        box_guard_child "$@" <&0 &
    CHILD_PID=$!
    CHILD_PGID=$(awk '{print $5}' "/proc/$CHILD_PID/stat" 2>/dev/null || echo "$CHILD_PID")
    if [[ "$CHILD_PGID" != "$CHILD_PID" ]]; then
        # setsid had to fork (should not happen from a script); fall back to the pid.
        CHILD_PGID=$CHILD_PID
    fi

    # 6. Watchdog.
    watchdog "$(gb_to_kb "$kill_below")" "$marker" "$minfile" "$rssfile" &
    WATCHDOG_PID=$!

    # 7. Wait for the child (wait returns early when a trapped signal arrives).
    local rc=0
    while :; do
        wait "$CHILD_PID"; rc=$?
        kill -0 "$CHILD_PID" 2>/dev/null || break
    done
    trap - INT TERM HUP
    # Reap anything left in the group (background grandchildren).
    kill_child_group TERM
    kill "$WATCHDOG_PID" 2>/dev/null; wait "$WATCHDOG_PID" 2>/dev/null
    local elapsed; elapsed=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN {printf "%.1f", b-a}')
    local oom_after; oom_after=$(oom_kill_count)
    local min_avail_seen="n/a"; [[ -s "$minfile" ]] && min_avail_seen="$(kb_to_gb "$(cat "$minfile")")GB"
    local peak_rss="n/a"; [[ -s "$rssfile" ]] && peak_rss="$(kb_to_gb "$(cat "$rssfile")")GB"
    local snap_after; snap_after=$(snapshot)
    log "after:  $snap_after"

    local final_rc=$rc verdict="ok"
    if [[ -s "$marker" ]]; then
        log "FAILED: watchdog killed the command: $(cat "$marker")"
        final_rc=86; verdict="watchdog_kill"
    fi
    if [[ "$oom_after" -gt "$oom_before" ]]; then
        log "FAILED: cgroup oom_kill rose ${oom_before} -> ${oom_after} during the run; results are invalid"
        [[ "$final_rc" -eq 0 ]] && final_rc=87
        verdict="${verdict}+oom_kill_rose"
    fi
    [[ -n "$SIGNALLED" ]] && verdict="${verdict}+signal_${SIGNALLED}"
    log "done: rc=$final_rc (child rc=$rc) elapsed=${elapsed}s lease_wait=${lease_wait}s min_avail=${min_avail_seen} peak_group_rss=${peak_rss} oom_kill ${oom_before}->${oom_after} verdict=$verdict"
    printf '%s pid=%s rc=%s child_rc=%s elapsed=%ss lease_wait=%ss min_avail=%s peak_group_rss=%s oom_kill=%s->%s verdict=%s label=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S')" "$$" "$final_rc" "$rc" "$elapsed" "$lease_wait" "$min_avail_seen" "$peak_rss" \
        "$oom_before" "$oom_after" "$verdict" "$label" >> "$LOG_FILE" 2>/dev/null || true
    : > "$HOLDER_FILE" 2>/dev/null || true
    rm -rf "$tmpd"
    return "$final_rc"
}

usage() {
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

check options: --min-avail-gb N (6) --need-disk-gb N (1) --max-load N (2)
               --max-gpu-mem-mib N (600) --max-gpu-util N (5) --quiet-s S (0: one snapshot)
run options:   --min-avail-gb N (6) --need-disk-gb N (1) --wait-min M (30) --poll-s S (30)
               --kill-below-gb G (3) --settle-s S (10) --max-gpu-mem-mib N (600)
               --max-gpu-util N (5) --label TEXT
EOF
}

main() {
    local sub=${1:-}; [[ $# -gt 0 ]] && shift
    case "$sub" in
        check) cmd_check "$@" ;;
        run) cmd_run "$@" ;;
        lease) lease_status ;;
        -h|--help|help|"") usage ;;
        *) die 2 "unknown subcommand '$sub' (check | run | lease)" ;;
    esac
}

main "$@"
