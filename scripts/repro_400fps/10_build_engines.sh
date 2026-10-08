#!/usr/bin/env bash
# Step 1: build an engine set from scratch, plus the TensorRT TAESD decoder engine.
#   scripts/repro_400fps/10_build_engines.sh [--set r5|r2] [--hardware-compat ampere_plus] [--dry-run]
#   --set r5 (default) -> models/tensorrt_unet_stagewise_sm<cc>_r5 (sm89 here; ≈400 fps; ~20-25 min; peak ~10.5 GB
#                         of MemAvailable)
#   --set r2           -> models/tensorrt_unet_stagewise_sm<cc>_r2 (350 fps; ~25 min)
#   --hardware-compat ampere_plus -> models/tensorrt_unet_stagewise_ampere_plus_<set> and the hardware-compatible TAESD
#       engine: TensorRT AMPERE_PLUS plans that load on every GPU of compute capability 8.0+ (the served default,
#       configs/trt_bundles/ampere-plus-r5-srcg50-int8.json; ~45 min: every tactic is timed without a seed cache).
#   MUSETALK_REPRO_ROOT overrides the root name.
#
# Both sets are the stagewise bs16 TensorRT UNet, variant "srccache" (conv_in + down0.resnets[0] precomputed per
# source frame, bit-identical to the full forward):
#   r5  FP16 prefix, down0rest, up3, tail; INT8 Q/DQ on the 117 layers of recipe gmac_0.50 in down1..up2
#       (Q/DQ only on those layers, input amax pinned to the recipe, weights per-channel max; other layers FP16)
#   r2  FP16 everywhere except down3 and mid, which are fully INT8 with modelopt INT8_DEFAULT_CFG max calibration on
#       8 corpus batches (main split) - the recipe of the published srcmix set
# The prefix block is built last. When the final block lands, the builder finalises the set: it loads the chain
# through the runtime backend, checks CUDA graph == direct and determinism, and records the probe hash.
#
# A rebuild is not byte-identical to the published engines. TensorRT tactic timing varies (the builder notes up to
# ~5% per block); the published blocks were built cold in separate roots, this script builds them in one root with
# one timing cache. The ONNX of every block must still match the published manifest (checked at the end); the
# engine hashes will differ. 20_gate.sh and 30_benchmark.sh re-measure accuracy and speed.
source "$(dirname "$0")/lib.sh"
SET=r5; DRY=0; HW=none
while [ $# -gt 0 ]; do case $1 in --set) SET=$2; shift 2 ;; --hardware-compat) HW=$2; shift 2 ;; --dry-run) DRY=1; shift ;;
  *) die "unknown arg $1" ;; esac; done
case $SET in r5) PUB=$PUBLISHED_R5 ;; r2) PUB=$PUBLISHED_R2 ;; *) die "--set r5|r2" ;; esac
# default builds are for this GPU's compute capability (sm89 on the RTX 4070 SUPER, sm86 on an RTX 3090)
case $HW in
  none) ARCH="sm$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '. ')"
        [[ "$ARCH" =~ ^sm[0-9]+$ ]] || die "cannot detect GPU compute capability; refusing native build" ;;
  ampere_plus) ARCH=ampere_plus ;;
  *) die "--hardware-compat none|ampere_plus" ;;
esac
[ -n "${MUSETALK_REPRO_ROOT:-}" ] || { ROOT_NAME="tensorrt_unet_stagewise_${ARCH}_$SET"; ENGINE_ROOT="models/$ROOT_NAME"; }
run() { if [ $DRY -eq 1 ]; then echo "  $*"; else "$@" || die "step failed: $*"; fi; }
B="$PY scripts/build_unet_stagewise.py --batch 16 --opt-level 5 --variant srccache --root $ENGINE_ROOT --calib-dir $CORPUS"
[ $HW = none ] || B="$B --hardware-compat $HW"
SEED=docs/fps_comparisons/4070s_300fps_20260927/unet_probe/tt16_timing_cache.bin
if [ "$HW" = none ] && [ "$ARCH" != sm89 ]; then
  B="$B --strict-timing-cache"
  log "native $ARCH: no 4070 timing seed; existing timing cache must match target metadata and SHA-256"
elif [ "$HW" = none ] && [ -e "$SEED" ]; then
  log "note: historical 4070 SUPER builds may seed from $SEED; other GPU models require strict cache provenance"
fi
if [ $DRY -eq 0 ] && [ -e "$ENGINE_ROOT/bs16/manifest.json" ] && \
   python3 -c "import json,sys; sys.exit(0 if json.load(open('$ENGINE_ROOT/bs16/manifest.json')).get('complete') else 1)"; then
  die "$ENGINE_ROOT is already complete; set MUSETALK_REPRO_ROOT to a new name, or remove it yourself"
fi
log "set $SET -> $ENGINE_ROOT"
if [ $SET = r5 ]; then
  run guarded build_${SET}_fp16_blocks 8 $B --blocks down0rest,up3,tail --report "$OUT/build_${SET}_fp16_blocks_report.json"
  run guarded build_${SET}_int8_blocks 14 $B --blocks down1,down2,down3,mid,up0,up1,up2 --int8-recipe "$RECIPE" \
      --report "$OUT/build_${SET}_int8_blocks_report.json"
else
  run guarded build_${SET}_fp16_blocks 10 $B --blocks down0rest,down1,down2,up0,up1,up2,up3,tail --report "$OUT/build_${SET}_fp16_blocks_report.json"
  run guarded build_${SET}_int8_blocks 14 $B --blocks down3,mid --int8-blocks down3,mid --calib-batches 8 \
      --report "$OUT/build_${SET}_int8_blocks_report.json"
fi
run guarded build_${SET}_prefix_finalize 8 $B --blocks prefix --report "$OUT/build_${SET}_prefix_finalize_report.json"
if [ $DRY -eq 0 ]; then
  python3 - "$ENGINE_ROOT/bs16/manifest.json" "$PUB/bs16/manifest.json" <<'EOF' || die "engine set did not finalise; see $OUT/build_*.log"
import json, os, sys
m = json.load(open(sys.argv[1]))
p = m.get("probe", {})
print(f"complete {m.get('complete')}  probe output_sha256 {p.get('output_sha256', '')[:16]}  "
      f"graph==direct {p.get('graph_equals_direct_enqueue')}  deterministic {p.get('deterministic_run_to_run')}")
if os.path.exists(sys.argv[2]):
    pub = json.load(open(sys.argv[2]))
    for b, e in m.get("blocks", {}).items():
        q = pub.get("blocks", {}).get(b, {})
        same = e.get("onnx_sha256") == q.get("onnx_sha256")
        print(f"  {b:10s} onnx {'MATCH' if same else 'DIFF '} vs published  precision {e.get('build_flags', {}).get('precision', '?')}")
    print(f"  published probe output_sha256 {pub.get('probe', {}).get('output_sha256', '')[:16]} (a rebuild's differs: FP16/INT8 tactics)")
sys.exit(0 if m.get("complete") else 1)
EOF
fi
# TensorRT TAESD decoder (bs8, fused uint8 post, optimisation level 3); reused when an engine with the same key exists
run guarded build_taesd_trt 8 env MUSETALK_TAESD_TRT_BATCH=8 MUSETALK_TAESD_TRT_HW_COMPAT=$HW "$PY" scripts/vae_fast_decoder.py build
log "done. Next: scripts/repro_400fps/20_gate.sh $ENGINE_ROOT"
