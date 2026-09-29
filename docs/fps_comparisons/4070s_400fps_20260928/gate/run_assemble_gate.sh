#!/usr/bin/env bash
# Assemble <name> = srcmix (prefix, down0rest, up3, tail) + recipe blocks from <overlay root>, finalize, run the repo UNet gate.
# usage: run_assemble_gate.sh <name> <overlay root> <blocks csv>
set -uo pipefail
R=/workspace/MuseTalk-perf300; cd $R; D=$R/docs/fps_comparisons/4070s_400fps_20260928/gate
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; C=calibration/unet_multi_avatar_20260928; G="scripts/box_guard.sh run --wait-min 90"
name=$1; ov=$2; blocks=$3; root=models/tensorrt_unet_stagewise_sm89_$name
$PY scripts/assemble_stagewise_set.py --base models/tensorrt_unet_stagewise_sm89_srcmix --overlay $ov:$blocks --out $root || exit 1
$G --min-avail-gb 8 --label finalize_$name -- $PY scripts/build_unet_stagewise.py --batch 16 --opt-level 5 --variant srccache --root $root --blocks prefix --report $D/finalize_${name}_report.json > $D/finalize_$name.log 2>&1; echo "$name finalize rc=$?"
for split in main holdout; do dir=$C; [[ $split == holdout ]] && dir=$C/holdout
  MUSETALK_UNET_BACKEND=trt_stagewise MUSETALK_UNET_STAGEWISE_BATCH=16 MUSETALK_UNET_STAGEWISE_CACHE_DIR=$root MUSETALK_TRT_FALLBACK=0 \
  $G --min-avail-gb 6 --label gunet_${name}_$split -- $PY scripts/validate_unet_backend.py --backend runtime --capture-dir $dir --padded-batch-size 8 --group-captures 2 --limit 0 --warmup 1 --iters 2 --fail-mae 0.01 --fail-max-abs 0.5 --report-path $D/gunet_${name}_$split.json > $D/gunet_${name}_$split.log 2>&1
  echo "$name gunet $split rc=$? $(python3 -c "import json;d=json.load(open('$D/gunet_${name}_$split.json'));s=d.get('summary',d);print({k:s[k] for k in s if 'mae' in k or 'abs' in k or 'pass' in k})" 2>/dev/null)"
done
