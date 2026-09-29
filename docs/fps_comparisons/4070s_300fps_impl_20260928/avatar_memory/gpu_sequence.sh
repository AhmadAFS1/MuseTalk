#!/usr/bin/env bash
# Avatar memory layout - OPTIONAL end-to-end GPU confirmation (plan gate E0).
#
# Nothing in the avatar memory layout needs the GPU: storage/loading is host-side, the
# latent cycle is untouched, and compose_frame() - the only consumer of the stored frames
# and masks - was proven bit-identical on CPU (gate_bitexact.json: 15 avatar/variant
# entries, every cycle position incl. wrap and mirror seams, 121k outputs, 0 mismatches).
# Decoded faces do not depend on the layout. This script re-proves it in the real
# HLSGPUStreamScheduler path with real TAESD faces, pre-encoder frames SHA-compared.
#
# Steps (each prints one PASS/FAIL line and writes JSON next to this script):
#   S0 preflight (CPU, <1 min)  gate_bitexact.json verdict must be PASS; harness present.
#   S1 golden_base (GPU)        replay_scheduler_exactness.py --mode golden, default layout
#                               (today). ~8-12 min incl. model load. Output golden_mem_base.json.
#                               Skipped with REUSE_SCHED_BASE=1 if ../scheduler/golden_base_A.json
#                               exists (the scheduler area's baseline, same worktree code).
#   S2 golden_layout (GPU)      same, with every layout flag on for the whole process:
#                               FRAME_STORE=png MASK_STORE=png MASK_CHANNELS=1 PLAN_FLOAT_ALPHA=0.
#                               ~8-12 min. Output golden_mem_layout.json.
#   S3 compare (CPU, <1 min)    --mode compare S1 vs S2. Threshold: identical=true for every
#                               job (frame, yuv420p and face SHA-256, order, status).
#                               Output golden_mem_compare.json. FAIL here with a nondeterministic
#                               baseline (scheduler area's base_r1 vs base_r2 not identical) is
#                               INCONCLUSIVE, not a layout defect.
#   S4 speed spot check (GPU, optional, RUN_SPEED=1) replay --mode speed --paced at N=16 for 60 s
#                               under the layout flags: generated fps and process CPU vs the
#                               scheduler area's default-layout speed run. ~5 min.
#
# RAM: the replay harness peaks at ~8.5-10.6 GB host RSS during model load, so every GPU
# step asks box_guard for --min-avail-gb 11. Expected total: ~20-25 min GPU (S1+S2), ~5 more
# with RUN_SPEED=1.
set -uo pipefail
cd /workspace/MuseTalk-perf300
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
HERE=docs/fps_comparisons/4070s_300fps_impl_20260928/avatar_memory
SCHED=docs/fps_comparisons/4070s_300fps_impl_20260928/scheduler
GUARD="scripts/box_guard.sh run --min-avail-gb 11 --wait-min 60"
LAYOUT_ENV=(MUSETALK_AVATAR_FRAME_STORE=png MUSETALK_AVATAR_MASK_STORE=png
            MUSETALK_AVATAR_MASK_CHANNELS=1 MUSETALK_AVATAR_PLAN_FLOAT_ALPHA=0)
STATUS=$HERE/gpu_sequence_status.json
declare -A RESULT

record() {  # step verdict detail
  RESULT[$1]="$2"
  echo "$1: $2 - $3"
  $PY - "$STATUS" "$1" "$2" "$3" <<'EOF'
import json, sys, time
path, step, verdict, detail = sys.argv[1:]
try:
    data = json.load(open(path))
except Exception:
    data = {}
data[step] = {"verdict": verdict, "detail": detail, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(data, open(path, "w"), indent=1)
EOF
}

# S0 ------------------------------------------------------------------------------------
if $PY -c "import json,sys; sys.exit(0 if json.load(open('$HERE/gate_bitexact.json'))['verdict']=='PASS' else 1)" \
   && [ -f scripts/replay_scheduler_exactness.py ]; then
  record S0_preflight PASS "CPU bit-exact gate PASS; replay harness present"
else
  record S0_preflight FAIL "gate_bitexact.json missing or not PASS - run verify_layout_bitexact.py first"
  exit 1
fi

# S1 ------------------------------------------------------------------------------------
BASE_JSON=$HERE/golden_mem_base.json
if [ "${REUSE_SCHED_BASE:-0}" = 1 ] && [ -f $SCHED/golden_base_A.json ]; then
  BASE_JSON=$SCHED/golden_base_A.json
  record S1_golden_base PASS "reused $BASE_JSON"
else
  $GUARD --label avatar_memory_S1_golden_base -- \
    $PY scripts/replay_scheduler_exactness.py --mode golden --run mem_base:head \
    --out $BASE_JSON > $HERE/golden_mem_base.log 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f $BASE_JSON ]; then
    record S1_golden_base PASS "rc=0 $(grep -c . $HERE/golden_mem_base.log) log lines -> $BASE_JSON"
  else
    record S1_golden_base FAIL "rc=$rc (75/76 = lease or RAM wait timed out; 86/87 = watchdog) see golden_mem_base.log"
    exit 1
  fi
fi

# S2 ------------------------------------------------------------------------------------
env "${LAYOUT_ENV[@]}" $GUARD --label avatar_memory_S2_golden_layout -- \
  env "${LAYOUT_ENV[@]}" $PY scripts/replay_scheduler_exactness.py --mode golden --run mem_layout:head \
  --out $HERE/golden_mem_layout.json > $HERE/golden_mem_layout.log 2>&1
rc=$?
layout_seen=$(grep -c "Avatar memory layout: frames=png masks=png/1ch plan_float_alpha=0" $HERE/golden_mem_layout.log)
if [ $rc -eq 0 ] && [ -f $HERE/golden_mem_layout.json ] && [ "$layout_seen" -gt 0 ]; then
  record S2_golden_layout PASS "rc=0, layout banner seen on $layout_seen avatar loads"
else
  record S2_golden_layout FAIL "rc=$rc layout_banner_loads=$layout_seen (0 = flags did not reach the loader)"
  exit 1
fi

# S3 ------------------------------------------------------------------------------------
$PY scripts/replay_scheduler_exactness.py --mode compare --compare $BASE_JSON $HERE/golden_mem_layout.json \
  --out $HERE/golden_mem_compare.json > $HERE/golden_mem_compare.log 2>&1
rc=$?
summary=$($PY - "$HERE/golden_mem_compare.json" <<'EOF'
import json, sys
reports = json.load(open(sys.argv[1]))["compare"]
frames = sum(r["frames_total"] for r in reports)
bad = sum(j.get("frame_mismatches", 0) + j.get("yuv_mismatches", 0) + j.get("face_mismatches", 0)
          for r in reports for j in r["jobs"].values())
print(f"frames={frames} mismatches={bad} identical={all(r['identical'] for r in reports)}")
EOF
)
if [ $rc -eq 0 ]; then
  record S3_compare PASS "$summary"
else
  record S3_compare FAIL "$summary (check scheduler base_r1 vs base_r2 determinism before blaming the layout)"
  exit 1
fi

# S4 (optional) -----------------------------------------------------------------------------
if [ "${RUN_SPEED:-0}" = 1 ]; then
  env "${LAYOUT_ENV[@]}" $GUARD --label avatar_memory_S4_speed_layout -- \
    env "${LAYOUT_ENV[@]}" $PY scripts/replay_scheduler_exactness.py --mode speed --paced --n-jobs 16 \
    --seconds 60 --run mem_layout_speed:head --out $HERE/speed_mem_layout.json > $HERE/speed_mem_layout.log 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    record S4_speed_layout PASS "see speed_mem_layout.json (compare fps/CPU with the scheduler area's default run)"
  else
    record S4_speed_layout FAIL "rc=$rc"
  fi
fi
echo "avatar_memory gpu_sequence done: ${!RESULT[*]} -> $STATUS"
