#!/usr/bin/env bash
# Step 4: labelled comparison videos for sign-off, from a 30_benchmark.sh V (and Q) capture. CPU only; ~5 min,
# ~2.5 GB RAM; it takes the GPU lease anyway, so it can never overlap an engine build.
#   scripts/repro_400fps/40_videos.sh [engine root] [r2 capture dir]
# Writes experiments/video_validation/<tag>_focus/: BEFORE | r2 | the set, at native 512 px, crf 10, with the diff row,
# metrics and verdicts. Every column is rebuilt from the exact rendered frames (hash-checked against the harness record)
# and encoded once, so no version gets a compression advantage.
# The r2 column comes from the published r2 capture (its raw arrays are gitignored). On a fresh machine, first
# build and capture r2: 10_build_engines.sh --set r2, then 30_benchmark.sh models/tensorrt_unet_stagewise_sm89_r2 Q V,
# then pass $OUT/chin_multistream/V_repro_r2 here.
# (scripts/video_signoff.py makes the plain sign-off reel of the PUBLISHED r2/r5 and rewrites signoff_r2_r5/.)
source "$(dirname "$0")/lib.sh"
ROOT="${1:-$ENGINE_ROOT}"
TAG=$(tag "$ROOT")
R2CAP="${2:-docs/fps_comparisons/4070s_300fps_impl_20260928/chin_multistream/V_srcmix_taesdtrt}"
R2Q=srcmix_taesdtrt; case "$R2CAP" in *V_repro_*) R2Q=${R2CAP##*/V_} ;; esac
CAP="$OUT/chin_multistream/V_$TAG"
[ -d "$CAP" ] || die "no V capture at $CAP; run 30_benchmark.sh $ROOT V first"
ls "$R2CAP"/stream00_*_faces.npz >/dev/null 2>&1 || die "r2 capture $R2CAP has no raw arrays; see the header for building r2"
FPS=$(grep -a '^SUMMARY' "$OUT/T_$TAG.log" 2>/dev/null | grep -o '"median_aggregate_fps": [0-9.]*' | awk '{printf "%.1f", $2}')
FPS="${FPS:-unmeasured}"
guarded "videos_$TAG" 6 env MUSETALK_QUALITY_RUNS="$QRUNS_OUT" "$PY" scripts/video_lineage.py --col-width 512 --crf 10 \
  --out-name "${TAG}_focus" \
  --arm "r2 (previous round)|350.2 fps aggregate (6 streams)|stagewise FP16 UNet + source-prefix cache + INT8 down3/mid + TRT TAESD|$R2CAP|$R2Q" \
  --arm "$TAG NEW|$FPS fps aggregate (6 streams)|rebuilt engine set $(basename "$ROOT")|$CAP|$TAG" || die "lineage videos failed"
grep -a '^wrote' "$OUT/videos_$TAG.log"
log "videos: experiments/video_validation/${TAG}_focus/"
