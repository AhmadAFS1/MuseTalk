#!/usr/bin/env bash
# r5 selection inputs: per-block times of every candidate set (interleaved), then per-block fake-quant errors.
set -uo pipefail
R=/workspace/MuseTalk-perf300; cd $R; D=$R/docs/fps_comparisons/4070s_400fps_20260928
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; G="scripts/box_guard.sh run --wait-min 90"
$G --min-avail-gb 6 --label bench_blocks_gmac -- $PY scripts/bench_stagewise_blocks.py --root models/tensorrt_unet_stagewise_sm89_srcmix \
   --root models/tensorrt_unet_stagewise_sm89_blkA_thr_8e-06 --root models/tensorrt_unet_stagewise_sm89_gmac_0.50 \
   --root models/tensorrt_unet_stagewise_sm89_gmac_0.55 --root models/tensorrt_unet_stagewise_sm89_gmac_0.59 --rounds 9 \
   --out $D/blocks/gmac_sets.json > $D/blocks/bench_gmac_sets.log 2>&1; echo "bench rc=$?"
$G --min-avail-gb 8 --label int8_study_perblock_sets -- $PY scripts/int8_layer_study.py --stage recipes --recipes $D/int8_study/recipes_perblock_sets.json \
   --out $D/int8_study/study.json > $D/int8_study/recipes_perblock_sets.log 2>&1; echo "perblock rc=$?"
