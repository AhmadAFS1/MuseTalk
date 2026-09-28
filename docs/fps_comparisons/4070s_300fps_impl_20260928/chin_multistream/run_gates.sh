#!/usr/bin/env bash
# Multi-stream chin render harness gates. Every step is its own box_guard lease (< 25 min each).
# Usage: run_gates.sh <step> [...]
#   E      exactness: baseline backends (compiled TAESD + shipping .ts bs8, --align stream8), N=6 x loops 1
#   Ta     throughput: baseline N=6, >= 60 s timed, 2 repeats (every clip is also hash-checked)
#   S      single-stream serial baseline (render_stage loop verbatim), all six identities x 3 runs
#   S1     harness with N=1 (baseline), 2 repeats
#   Tb6 / Tb12   stagewise bs16 + compiled TAESD, N=6 / N=12 (6 identities x 2), 2 repeats
#   Tb12x3       N=12 over 3 identities x 4 streams (1.8 GB less arena RAM; fallback when RAM is short)
#   Cb     stagewise bs16 + compiled TAESD candidate outputs: N=6 x loops 1, crf 12 videos + arrays + accepted compare
#   Tc6 / Tc12 / Cc   same with TRT TAESD (only if its gates passed)
#   R6 / R12   CPU-only replay ceiling (accepted faces, no GPU), N=6 / N=12
# RAM: the .ts UNet load has a ~8 GB transient peak (bench_gpu_path baseline: VmHWM 9.5 GB), so
# baseline steps need more headroom than stagewise ones. Arenas: ~0.6 GB per identity (/dev/shm).
set -uo pipefail
R=/workspace/MuseTalk-perf300
O=$R/docs/fps_comparisons/4070s_300fps_impl_20260928/chin_multistream
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
H="$PY $R/scripts/chin_multistream_render.py"
G="$R/scripts/box_guard.sh run --wait-min 90"
cd "$R"
run() {  # run <label> <min-avail-gb> <args...>
  local label=$1 avail=$2; shift 2
  $G --min-avail-gb "$avail" --label "chinms_$label" -- $H --label "$label" "$@" > "$O/$label.log" 2>&1
  local rc=$?
  echo "=== $label rc=$rc $(date -u +%H:%M:%S) $(grep -a '^SUMMARY' "$O/$label.log" | tail -1 | cut -c1-400)"
  return $rc
}
for step in "$@"; do
  echo "=== step $step $(date -u +%H:%M:%S)"
  case "$step" in
    E)    run gateE_baseline_n6_loops1 11.5 --backend baseline --streams 6 --loops 1 ;;
    Ta)   run Ta_baseline_n6 11.5 --backend baseline --streams 6 --loops 12 --repeats 2 --min-timed-s 60 ;;
    S)    run S_serial_baseline 11.5 --mode serial --backend baseline --repeats 3 ;;
    S1)   run S1_harness_baseline_n1 11.5 --backend baseline --streams 1 --identities black_woman --loops 8 --repeats 2 ;;
    Tb6)  run Tb_stagewise16_n6 10 --backend stagewise16 --streams 6 --loops 14 --repeats 2 --min-timed-s 60 ;;
    Tb12) run Tb_stagewise16_n12 12 --backend stagewise16 --streams 12 --loops 7 --repeats 2 --min-timed-s 60 ;;
    Tb12x3) run Tb_stagewise16_n12_3ids 10.5 --backend stagewise16 --streams 12 --identities black_woman,east_asian_man_goatee,middle_eastern_man_full_beard --loops 7 --repeats 2 --min-timed-s 60 ;;
    Cb)   run Cb_stagewise16_n6_outputs 11.5 --backend stagewise16 --streams 6 --loops 1 --encode --crf 12 --save-arrays --compare-accepted ;;
    Tc6)  run Tc_stagewise16_taesdtrt_n6 10 --backend stagewise16_taesdtrt --streams 6 --loops 14 --repeats 2 --min-timed-s 60 ;;
    Tc12) run Tc_stagewise16_taesdtrt_n12 12 --backend stagewise16_taesdtrt --streams 12 --loops 7 --repeats 2 --min-timed-s 60 ;;
    Cc)   run Cc_stagewise16_taesdtrt_n6_outputs 11.5 --backend stagewise16_taesdtrt --streams 6 --loops 1 --encode --crf 12 --save-arrays --compare-accepted ;;
    R6)   run R_replay_n6 9.5 --backend replay --streams 6 --loops 6 --repeats 1 ;;
    R12)  run R_replay_n12 10.5 --backend replay --streams 12 --loops 3 --repeats 1 ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
done
