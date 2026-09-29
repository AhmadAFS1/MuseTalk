#!/usr/bin/env bash
# Round 2 recipe engines: every layer of down1..up2 except the sensitivity blacklist and the audio K/V
# projections (attn2.to_k/to_v clip on unseen real speech); up3/down0 stay FP16.
set -uo pipefail
R=/workspace/MuseTalk-perf300; cd $R; D=$R/docs/fps_comparisons/4070s_400fps_20260928
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; G="scripts/box_guard.sh run --wait-min 90"
for r in "$@"; do
  root=models/tensorrt_unet_stagewise_sm89_$r
  $G --min-avail-gb 8 --label build_$r -- $PY scripts/build_unet_stagewise.py --batch 16 --opt-level 5 --variant srccache \
     --root $root --blocks down1,down2,down3,mid,up0,up1,up2 --int8-recipe $D/int8_study/recipe_$r.json \
     > $D/blocks/build_$r.log 2>&1
  echo "build $r rc=$?"
done
