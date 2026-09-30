#!/usr/bin/env bash
# Step 2: correctness gates for an engine set (default: the one 10_build_engines.sh made).
#   scripts/repro_400fps/20_gate.sh [engine root]      e.g. models/tensorrt_unet_stagewise_sm89_srcg50 (the published r5 set)
#
# The gates record results; they do not decide. Published values (repo UNet gate: mae_max <= 0.01, max_abs <= 0.5):
#   r5  main 0.0044 / 0.78, holdout 0.0039 / 1.26  -> mae passes, max_abs fails. That is expected for INT8 with this
#       coverage; r5 was judged on pixels and video instead (experiments/video_validation/r5_*/README.md).
#   r2  main 0.0025 / 0.39, holdout 0.0021 / 0.24  -> passes.
#   TAESD (the published G-TAESD gate, 3584 frames): max 5 LSB vs the 3 LSB bar (FAIL), mean 0.066 (passes);
#       used from r2 on with the user's decision pending.
source "$(dirname "$0")/lib.sh"
ROOT="${1:-$ENGINE_ROOT}"
[ -e "$ROOT/bs16/manifest.json" ] || die "no engine set at $ROOT"
T=$(tag "$ROOT")
HW=$(hw_compat_of "$ROOT")   # a hardware-compatible set is gated with the hardware-compatible TAESD engine
E="MUSETALK_UNET_BACKEND=trt_stagewise MUSETALK_UNET_STAGEWISE_BATCH=16 MUSETALK_UNET_STAGEWISE_CACHE_DIR=$ROOT MUSETALK_TRT_FALLBACK=0"
for split in main holdout; do
  dir=$CORPUS; [ $split = holdout ] && dir=$CORPUS/holdout
  rep="$OUT/gate_unet_${T}_$split.json"
  guarded "gate_unet_${T}_$split" 8 env $E "$PY" scripts/validate_unet_backend.py --backend runtime --capture-dir "$dir" \
    --padded-batch-size 8 --group-captures 2 --limit 0 --warmup 1 --iters 2 --fail-mae 0.01 --fail-max-abs 0.5 --report-path "$rep"
  if [ -s "$rep" ]; then
    python3 -c "import json; d=json.load(open('$rep')); s=d.get('summary',d); print('  UNet gate $split: mae_max %.4f (<= 0.01: %s)  max_abs %.3f (<= 0.5: %s)' % (s['mae_max'], s['mae_max']<=0.01, s['max_abs_max'], s['max_abs_max']<=0.5))"
  else
    log "gate_unet_${T}_$split: no report written; see $OUT/gate_unet_${T}_$split.log"
  fi
done
# forward_cached (precomputed source prefix) must equal the full forward bit for bit, including permuted rows
guarded "gate_srccache_${T}" 8 "$PY" "$PKG/srccache_exact.py" --root "$ROOT" --corpus "$CORPUS" --out "$OUT/gate_srccache_${T}.json"
grep -a '^PASS\|^FAIL' "$OUT/gate_srccache_${T}.log" | cut -c1-220 || log "srccache gate: no verdict; see its log"
# TAESD: load + probe check (never builds), then the published G-TAESD gate itself. The verdict is recorded into the
# engine meta only when the engine has none yet (a fresh engine); an existing record is left as it is.
guarded "gate_taesd_load_${T}" 6 env MUSETALK_TAESD_TRT_BATCH=8 MUSETALK_TAESD_TRT_BUILD=0 MUSETALK_TAESD_TRT_HW_COMPAT=$HW \
  "$PY" scripts/vae_fast_decoder.py verify
L=$(grep -a 'TAESD TRT backend' "$OUT/gate_taesd_load_${T}.log" | tail -1)
echo "  $L"
REC=--no-record   # record only when the loader says this key has no verdict yet (a freshly built engine)
case "$L" in *gate=PASS*|*gate=FAIL*|"") ;; *gate=*) REC= ;; esac
guarded "gate_taesd_${T}" 8 env REPRO_GATE_OUT="$OUT/taesd_gate" MUSETALK_TAESD_TRT_BATCH=8 MUSETALK_TAESD_TRT_BUILD=0 \
  MUSETALK_TAESD_TRT_HW_COMPAT=$HW \
  "$PY" "$PKG/gate_taesd_trt.py" $REC
python3 -c "import json; g=json.load(open('$OUT/taesd_gate/gate_taesd_trt.json'))['gate']; print('  G-TAESD: verdict %s  max %s LSB (bar 3)  mean %.3f (bar 0.2)  rows104 max %s' % (g['verdict'], g['G_TAESD_full_max'], g['G_TAESD_full_mean'], g['G_TAESD_rows104_max']))" 2>/dev/null \
  || log "G-TAESD: see $OUT/gate_taesd_${T}.log"
log "done. Next: scripts/repro_400fps/30_benchmark.sh $ROOT"
