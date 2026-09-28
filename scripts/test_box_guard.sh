#!/usr/bin/env bash
# Self-test for scripts/box_guard.sh (no GPU work; about 45 s).
#   scripts/test_box_guard.sh               private lease file (never blocks other agents)
#   scripts/test_box_guard.sh --real-lease  use /workspace/.gpu_lease
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
G="$HERE/box_guard.sh"
T=$(mktemp -d "${TMPDIR:-/tmp}/box_guard_test.XXXXXX")
trap 'rm -rf "$T"' EXIT
if [[ "${1:-}" != "--real-lease" ]]; then
    export BOX_GUARD_LEASE_FILE="$T/lease" BOX_GUARD_LOG="$T/lease.log"
fi
pass=0; fail=0
expect() {  # expect <name> <expected rc> <actual rc>
    if [[ "$2" == "$3" ]]; then echo "PASS $1 (rc=$3)"; pass=$((pass + 1)); else echo "FAIL $1 (expected rc=$2, got $3)"; fail=$((fail + 1)); fi
}
Q="--settle-s 1 --poll-s 2"

out=$("$G" run $Q --label t_trivial -- bash -c 'echo "oom_score_adj=$(cat /proc/self/oom_score_adj) pgid=$(ps -o pgid= $$ | tr -d " ") pid=$$"' 2>/dev/null); rc=$?
echo "  child: $out"
pg=$(sed -n 's/.*pgid=\([0-9]*\).*/\1/p' <<< "$out"); pd=$(sed -n 's/.* pid=\([0-9]*\).*/\1/p' <<< "$out")
[[ "$out" == *"oom_score_adj=1000"* && -n "$pg" && "$pg" == "$pd" ]]; expect "trivial command, oom_score_adj=1000, own process group" 0 $(( rc + $? ))

"$G" run $Q --label t_rc -- bash -c 'exit 3' 2>/dev/null; expect "child exit code passthrough" 3 $?

# Serialization: B must start only after A ends.
( "$G" run $Q --label t_holderA -- bash -c 'date +%s.%N > '"$T"'/a_start; sleep 8; date +%s.%N > '"$T"'/a_end' 2>/dev/null ) &
sleep 2
"$G" run $Q --wait-min 1 --label t_waiterB -- bash -c 'date +%s.%N > '"$T"'/b_start' 2>"$T/b.err"; rcb=$?
wait
ok=$(awk -v ae="$(cat "$T/a_end")" -v bs="$(cat "$T/b_start")" 'BEGIN {print (bs > ae) ? 0 : 1}')
echo "  A ended $(cat "$T/a_end"), B started $(cat "$T/b_start"); B log: $(grep -c 'waiting for GPU lease' "$T/b.err") waiting lines"
expect "second concurrent invocation serialized behind the lease" 0 $(( rcb + ok ))

( "$G" run $Q --label t_holderC -- sleep 12 2>/dev/null ) &
sleep 2
"$G" run $Q --wait-min 0.1 --label t_impatient -- echo SHOULD_NOT_RUN 2>/dev/null; expect "lease wait timeout" 75 $?
wait

"$G" run $Q --kill-below-gb 100000 --label t_watchdog -- sleep 30 2>/dev/null; expect "watchdog kills group when MemAvailable < threshold" 86 $?

cp /sys/fs/cgroup/memory.events "$T/events" 2>/dev/null || printf 'oom 0\noom_kill 0\n' > "$T/events"
BOX_GUARD_MEMORY_EVENTS="$T/events" "$G" run $Q --label t_oom -- bash -c "sleep 1; awk '\$1==\"oom_kill\" {\$2=\$2+1} {print}' $T/events > $T/e2 && cp $T/e2 $T/events" 2>/dev/null
expect "oom_kill rise invalidates the run (simulated memory.events)" 87 $?

"$G" run $Q --min-avail-gb 100000 --wait-min 0.05 --label t_ram -- echo SHOULD_NOT_RUN 2>/dev/null; expect "MemAvailable below --min-avail-gb" 76 $?
"$G" run $Q --need-disk-gb 100000000 --label t_disk -- echo SHOULD_NOT_RUN 2>/dev/null; expect "disk below --need-disk-gb" 77 $?

( "$G" run $Q --label t_signal -- bash -c 'sleep 101 & sleep 101 & wait' 2>/dev/null; echo $? > "$T/sig_rc" ) &
sleep 4
gpid=$(awk -F'[ =]' '/label=t_signal/ {print $2}' "${BOX_GUARD_LEASE_FILE:-/workspace/.gpu_lease}.holder")
kill -TERM "$gpid"; wait
left=$(pgrep -x sleep -a | grep -c 'sleep 101')
expect "SIGTERM to the guard terminates the child group (leftover=$left)" 143 "$(( $(cat "$T/sig_rc") + left ))"

"$G" check --max-load -1 >/dev/null 2>&1; expect "check exits non-zero when the box is not quiet (--max-load -1)" 1 $?

echo "RESULT: $pass passed, $fail failed"
[[ $fail -eq 0 ]]
