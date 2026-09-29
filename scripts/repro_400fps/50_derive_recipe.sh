#!/usr/bin/env bash
# Optional: re-derive the r5 INT8 layer recipe from scratch (fake-quant study, no engines). About 10 min of GPU,
# ~4 GB host RAM. The published recipe (int8_study/recipe_gmac_0.50.json) is already in the repo; run this only to
# re-check the method or to derive a recipe for a different model / corpus.
#   scripts/repro_400fps/50_derive_recipe.sh [study json]        default: a fresh $OUT/study.json
#
# Method (docs/fps_comparisons/4070s_400fps_20260928/README.md §2-§3, §8.4-8.5):
#   1. inventory + baseline + per-layer MSE-optimal clip + single-layer sensitivity (max and mse clips)
#   2. greedy recipes by single-layer error per MAC (make_gmac_recipes.py), evaluated on main + holdout
#   3. export the chosen recipe for build_unet_stagewise.py --int8-recipe
# A fresh study can differ from the published one by a few layers near the cut (cudnn.benchmark picks kernels by
# timing, and the INT8 error sits near FP16 rounding). Compare the exported layer set with the published recipe.
# Choosing the MAC cut uses measured per-block engine times. For each candidate recipe, build a set with
# 10_build_engines.sh (MUSETALK_REPRO_RECIPE=<recipe> MUSETALK_REPRO_ROOT=<name>), time the sets with scripts/bench_stagewise_blocks.py
# and pick the lowest-error block combination under the 400 fps budget with select_combo.py (the budget is ~31.45 ms
# of UNet per bs16 call on this GPU with TRT TAESD; r5 = all blocks from gmac_0.50, 31.20 ms).
source "$(dirname "$0")/lib.sh"
STUDY="${1:-$OUT/study.json}"
S=docs/fps_comparisons/4070s_400fps_20260928/int8_study
case "$(realpath -m "$STUDY")" in "$(realpath -m "$S")"/*) die "refusing to rewrite the published study under $S; pass a path under $OUT" ;; esac
guarded study_sens 8 "$PY" scripts/int8_layer_study.py --stage inventory --stage baseline --stage amax_mse --stage sens --out "$STUDY" \
  || die "study failed"
python3 "$S/make_gmac_recipes.py" --study "$STUDY" --out "$OUT/recipes_gmac.json" || die "recipe selection failed"
guarded study_recipes 8 "$PY" scripts/int8_layer_study.py --stage recipes --recipes "$OUT/recipes_gmac.json" --out "$STUDY" \
  || die "recipe evaluation failed"
grep '^recipe' "$OUT/study_recipes.log"
for r in gmac_0.50 gmac_0.55 gmac_0.59; do
  "$PY" scripts/int8_layer_study.py --stage export --export-recipe $r --export-out "$OUT/recipe_$r.json" --out "$STUDY"
done
python3 -c "import json; a=json.load(open('$OUT/recipe_gmac_0.50.json'))['layers']; b=json.load(open('$S/recipe_gmac_0.50.json'))['layers']; print('layers: fresh %d, published %d, common %d' % (len(a), len(b), len(set(a) & set(b))))"
log "exported $OUT/recipe_gmac_*.json"
