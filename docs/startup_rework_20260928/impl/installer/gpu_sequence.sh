#!/usr/bin/env bash
# gpu_sequence.sh - validation the installer component could NOT run on the shared box
# (startup rework 2026-09-28, component C: scripts/install_musetalk.sh, requirements/*,
# download_weights.sh additions, scripts/musetalk_selftest.py).
#
# Steps that import torch or touch the GPU run under the GPU lease:
#   scripts/box_guard.sh run --min-avail-gb N --wait-min 60 --label startup_<step> -- <cmd>
# CPU-only steps (unit tests, the read-only --check) are cheap and run directly.
# Every step prints one "PASS <step>: ..." or "FAIL <step>: ..." line and appends a JSON line to
# $OUT/gpu_sequence_results.jsonl; artefacts land next to this script.
#
#   step                          GPU  expected runtime        peak host RAM        notes
#   ----------------------------  ---  ----------------------  -------------------  ------------------------------
#   startup_installer_unit_tests  no   ~15 s                   < 200 MB             scripts/test_install_musetalk.sh
#   startup_install_check         no   < 2 s                   < 100 MB             read-only --check of the live venv
#   startup_import_smoke          no*  30-60 s                 ~1.5-2 GB            --check --check-imports (CUDA hidden)
#   startup_selftest              yes  4-8 min (TAESD max-     ~6 GB (UNet mmap +   TAESD compile+timing, TAESD TRT
#                                      autotune compile)       fp16 copy), VRAM ~3  timing, eager UNet bs8, engine keys
#   startup_cu128_resolve         no   2-3 min                 ~250 MB pip, but     OPTIONAL (RUN_CU128_RESOLVE=1): final
#                                                              5-7 GB page cache    cu128 file dry-run on a quiet box
#   startup_fresh_install         no** 12-20 min + selftest    ~3 GB (pip), then    OPTIONAL (RUN_FULL_INSTALL=1): clean
#                                                              selftest as above    install into a scratch venv, needs
#                                                                                   >= 24 GB free disk and network
#   (* imports torch/tensorrt, so it takes the lease to respect the RAM floor; ** its self-test uses the GPU)
#
# Usage:  bash docs/startup_rework_20260928/impl/installer/gpu_sequence.sh
#         STEPS=startup_selftest bash .../gpu_sequence.sh          # run a subset (comma list)
#         RUN_FULL_INSTALL=1 RUN_CU128_RESOLVE=1 bash .../gpu_sequence.sh
set -Eeuo pipefail

REPO="${REPO:-/workspace/MuseTalk-perf300}"
OUT="${OUT:-$REPO/docs/startup_rework_20260928/impl/installer}"
VENV="${VENV:-/workspace/.venvs/musetalk_trt_stagewise}"
VENV_PY="$VENV/bin/python"
GUARD="${GUARD:-$REPO/scripts/box_guard.sh}"
STEPS="${STEPS:-all}"
RUN_FULL_INSTALL="${RUN_FULL_INSTALL:-0}"
RUN_CU128_RESOLVE="${RUN_CU128_RESOLVE:-0}"
SCRATCH="${SCRATCH:-/workspace/.startup_installer_scratch}"
RESULTS="$OUT/gpu_sequence_results.jsonl"
# Thresholds for the self-test verdict (RTX 4070 SUPER reference: TAESD compiled ~6.3-6.8 ms/bs8,
# TRT TAESD ~3.9 ms/bs8, eager UNet ~37 ms/bs8; generous margins).
MAX_TAESD_MS="${MAX_TAESD_MS:-15}"
MAX_TAESD_TRT_MS="${MAX_TAESD_TRT_MS:-10}"
MAX_UNET_EAGER_MS="${MAX_UNET_EAGER_MS:-60}"
EXPECTED_UNET_TS_KEY="${EXPECTED_UNET_TS_KEY:-sm89-nvidia-geforce-rtx-4070-super-trt10.3.0-tt2.5.0}"

FAILS=0
log() { printf '[gpu_sequence installer %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
want() { [[ "$STEPS" == "all" || ",$STEPS," == *",$1,"* ]]; }
record() {  # record STEP PASS|FAIL DETAIL [JSON_ARTEFACT]
  local step="$1" verdict="$2" detail="$3" artefact="${4:-}"
  printf '%s %s: %s\n' "$verdict" "$step" "$detail"
  if [[ "$verdict" == "FAIL" ]]; then FAILS=$((FAILS + 1)); fi
  python3 - "$RESULTS" "$step" "$verdict" "$detail" "$artefact" <<'PY'
import json, sys, time
path, step, verdict, detail, artefact = sys.argv[1:6]
with open(path, "a") as fh:
    fh.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "step": step,
                         "verdict": verdict, "detail": detail, "artefact": artefact or None}) + "\n")
PY
}
guarded() {  # guarded STEP MIN_AVAIL_GB cmd...
  local step="$1" min_avail="$2"; shift 2
  "$GUARD" run --min-avail-gb "$min_avail" --wait-min 60 --label "$step" -- "$@"
}

mkdir -p "$OUT"
cd "$REPO"
log "repo=$REPO venv=$VENV out=$OUT steps=$STEPS"

# ------------------------------------------------------------------ startup_installer_unit_tests
if want startup_installer_unit_tests; then
  status=0
  bash "$REPO/scripts/test_install_musetalk.sh" >"$OUT/unit_tests.log" 2>&1 || status=$?
  summary="$(tail -n 1 "$OUT/unit_tests.log")"
  if (( status == 0 )); then record startup_installer_unit_tests PASS "$summary" "$OUT/unit_tests.log"
  else record startup_installer_unit_tests FAIL "exit $status: $summary" "$OUT/unit_tests.log"; fi
fi

# ------------------------------------------------------------------ startup_install_check
if want startup_install_check; then
  status=0
  bash "$REPO/scripts/install_musetalk.sh" --check --venv "$VENV" --report "$OUT/check_live_venv.json" \
    >"$OUT/check_live_venv.log" 2>&1 || status=$?
  if (( status == 0 )); then record startup_install_check PASS "live venv passes --check (exit 0)" "$OUT/check_live_venv.json"
  else record startup_install_check FAIL "exit $status (10 = clean install, 11 = repair); see check_live_venv.log" "$OUT/check_live_venv.json"; fi
fi

# ------------------------------------------------------------------ startup_import_smoke
if want startup_import_smoke; then
  status=0
  guarded startup_import_smoke 5 \
    bash "$REPO/scripts/install_musetalk.sh" --check --check-imports --venv "$VENV" \
    >"$OUT/import_smoke.log" 2>&1 || status=$?
  if (( status == 0 )); then record startup_import_smoke PASS "torch/tensorrt/torch_tensorrt/diffusers/aiortc/av/... import with CUDA hidden" "$OUT/import_smoke.log"
  else record startup_import_smoke FAIL "exit $status (75/76 = box_guard wait/RAM; 11 = import failure); see import_smoke.log" "$OUT/import_smoke.log"; fi
fi

# ------------------------------------------------------------------ startup_selftest
judge_selftest() {  # judge_selftest JSON STEP
  python3 - "$1" "$MAX_TAESD_MS" "$MAX_TAESD_TRT_MS" "$MAX_UNET_EAGER_MS" "$EXPECTED_UNET_TS_KEY" <<'PY'
import json, sys
path, max_taesd, max_trt, max_unet, expected_key = sys.argv[1], *map(float, sys.argv[2:5]), sys.argv[5]
d = json.load(open(path))
problems = []
if d.get("schema") != "musetalk_gpu_selftest_v1": problems.append(f"schema {d.get('schema')}")
if not d.get("cuda_ok"): problems.append(f"cuda_ok false ({d.get('cuda_error')})")
if not d.get("trt_import_ok"): problems.append(f"trt_import_ok false ({d.get('trt_import_error')})")
t = d.get("taesd") or {}
if t.get("compile_ok") is not True: problems.append(f"taesd.compile_ok {t.get('compile_ok')} ({t.get('error')})")
if not t.get("ms_bs8") or t["ms_bs8"] > max_taesd: problems.append(f"taesd.ms_bs8 {t.get('ms_bs8')} > {max_taesd}")
if not t.get("eager_ms_bs8"): problems.append("taesd.eager_ms_bs8 missing")
u = d.get("unet_eager") or {}
if u.get("ok") is not True or not u.get("ms_bs8") or u["ms_bs8"] > max_unet: problems.append(f"unet_eager {u}")
trt = d.get("taesd_trt") or {}
if trt.get("engine_present") and (trt.get("ok") is not True or not trt.get("ms_bs8") or trt["ms_bs8"] > max_trt):
    problems.append(f"taesd_trt present but ok={trt.get('ok')} ms_bs8={trt.get('ms_bs8')} error={trt.get('error')}")
key = ((d.get("engine_keys") or {}).get("keys") or {}).get("unet_ts")
gpu = (d.get("gpu") or {}).get("name", "")
if "4070 SUPER" in gpu and key != expected_key: problems.append(f"unet_ts key {key} != {expected_key}")
summary = (f"gpu={gpu} taesd compiled {t.get('ms_bs8')} ms/bs8 (eager {t.get('eager_ms_bs8')}, warmup {t.get('warmup_s')} s), "
           f"taesd_trt {trt.get('ms_bs8')} ms/bs8 ({trt.get('layout')}), unet eager {u.get('ms_bs8')} ms/bs8, "
           f"keys={((d.get('engine_keys') or {}).get('keys'))}, est={((d.get('estimate') or {}).get('gpu_path_fps_eager_unet'))} fps")
print(("OK " if not problems else "BAD ") + summary + ("" if not problems else " | problems: " + "; ".join(problems)))
sys.exit(0 if not problems else 1)
PY
}

if want startup_selftest; then
  status=0
  # TORCHINDUCTOR_COMPILE_THREADS caps inductor's compile-worker pool (default min(32, nproc) processes,
  # each importing torch) - it changes compile parallelism only, never the generated kernels' results.
  guarded startup_selftest 9 env TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}" \
    "$VENV_PY" -B "$REPO/scripts/musetalk_selftest.py" --repo-root "$REPO" --out "$OUT/gpu_selftest.json" \
      --buckets 8 --unet >"$OUT/gpu_selftest.log" 2>&1 || status=$?
  if [[ -f "$OUT/gpu_selftest.json" ]] && verdict="$(judge_selftest "$OUT/gpu_selftest.json")"; then
    record startup_selftest PASS "$verdict (selftest exit $status)" "$OUT/gpu_selftest.json"
  else
    record startup_selftest FAIL "selftest exit $status; ${verdict:-no JSON written}; see gpu_selftest.log" "$OUT/gpu_selftest.json"
  fi
fi

# ------------------------------------------------------------------ startup_cu128_resolve (optional)
if want startup_cu128_resolve && [[ "$RUN_CU128_RESOLVE" == "1" ]]; then
  status=0
  rv="$SCRATCH/resolve_venv"
  guarded startup_cu128_resolve 12 bash -c "
    set -Eeuo pipefail
    rm -rf '$rv'; python3.10 -m venv '$rv'; '$rv/bin/python' -m pip install -q pip==26.2.1
    '$rv/bin/python' -m pip install --dry-run --ignore-installed --no-cache-dir --quiet --report '$OUT/resolve_cu128_final_report.json' \
      -r requirements/server.in -r requirements/kokoro.in -c requirements/constraints-cu128.txt \
      --extra-index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.nvidia.com
  " >"$OUT/resolve_cu128_final.log" 2>&1 || status=$?
  rm -rf "$rv"
  if (( status == 0 )); then record startup_cu128_resolve PASS "final constraints-cu128.txt resolves with both indexes" "$OUT/resolve_cu128_final_report.json"
  else record startup_cu128_resolve FAIL "exit $status; see resolve_cu128_final.log" "$OUT/resolve_cu128_final.log"; fi
fi

# ------------------------------------------------------------------ startup_fresh_install (optional)
if want startup_fresh_install && [[ "$RUN_FULL_INSTALL" == "1" ]]; then
  free_gb="$(df -P -BG /workspace | awk 'NR==2 {gsub("G","",$4); print $4}')"
  if (( free_gb < 24 )); then
    record startup_fresh_install FAIL "skipped: ${free_gb} GB free < 24 GB needed (venv ~9.5 GB + pip downloads)"
  else
    status=0
    mkdir -p "$SCRATCH"
    # Scratch venv, stamp, self-test and native VP8 dir: nothing touches the live venv or <repo>/.runtime
    # (in this worktree .runtime is a symlink into /workspace/MuseTalk). --skip-weights: models/ is already
    # validated by startup_install_check.
    guarded startup_fresh_install 8 env PIP_CACHE_DIR="$SCRATCH/pip-cache" WEBRTC_NATIVE_VP8_DIR="$SCRATCH/native_vp8" \
      bash "$REPO/scripts/install_musetalk.sh" --venv "$SCRATCH/venv" --skip-apt --skip-weights \
        --state-file "$SCRATCH/install_state.json" --selftest-out "$OUT/fresh_install_gpu_selftest.json" \
      >"$OUT/fresh_install.log" 2>&1 || status=$?
    diff_status=0
    if (( status == 0 )); then
      "$SCRATCH/venv/bin/python" -m pip freeze >"$OUT/fresh_install_freeze.txt"
      # Same version for every package both venvs have, and no package the validated venv lacks.
      # (The validated venv also carries avatar-prep/legacy-int8 packages a default install skips:
      # those are reported as "not installed", not as failures.)
      python3 - "$REPO/docs/startup_rework_20260928/venv_freeze_cu121.txt" "$OUT/fresh_install_freeze.txt" \
        "$OUT/fresh_install_freeze_compare.json" <<'PY' || diff_status=$?
import json, re, sys
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
def load(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if "==" in line:
            name, version = line.split("==", 1); out[norm(name)] = version
        elif " @ " in line:
            out[norm(line.split(" @ ", 1)[0])] = "URL"
    return out
validated, fresh = load(sys.argv[1]), load(sys.argv[2])
mismatch = {n: [validated[n], fresh[n]] for n in fresh if n in validated and fresh[n] != validated[n]}
new = sorted(n for n in fresh if n not in validated)
not_installed = sorted(n for n in validated if n not in fresh)
json.dump({"version_mismatch": mismatch, "new_packages": new, "not_installed_by_default_groups": not_installed},
          open(sys.argv[3], "w"), indent=1)
print(f"common={len(set(fresh) & set(validated))} mismatch={len(mismatch)} new={len(new)} not_installed={len(not_installed)}")
sys.exit(0 if not mismatch and not new else 1)
PY
    fi
    if (( status == 0 && diff_status == 0 )); then
      record startup_fresh_install PASS "clean install matches the validated freeze for every shared package (fresh_install_freeze_compare.json); self-test in fresh_install_gpu_selftest.json" "$OUT/fresh_install_freeze_compare.json"
    elif (( status == 0 )); then
      record startup_fresh_install FAIL "install ok but versions differ from the validated venv (fresh_install_freeze_compare.json)" "$OUT/fresh_install_freeze_compare.json"
    else
      record startup_fresh_install FAIL "install exit $status; see fresh_install.log" "$OUT/fresh_install.log"
    fi
    log "Scratch venv kept at $SCRATCH/venv for inspection; remove with: rm -rf '$SCRATCH'"
  fi
fi

log "results appended to $RESULTS"
if (( FAILS > 0 )); then
  log "$FAILS step(s) FAILED in this run"
  exit 1
fi
log "all selected steps PASSED"
