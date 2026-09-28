#!/usr/bin/env bash
set -uo pipefail
R=/workspace/MuseTalk-perf300; cd $R; D=$R/docs/fps_comparisons/4070s_300fps_impl_20260928/unet_fp16
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; C=calibration/unet_multi_avatar_20260928; G="scripts/box_guard.sh run --wait-min 90"
gate() { name=$1; root=models/tensorrt_unet_stagewise_sm89_$name
  $G --min-avail-gb 6 --label srccache_exact_$name -- $PY $D/srccache_exact.py --root $root > $D/srccache_exact_$name.log 2>&1; echo "$name exact rc=$?"
  $G --min-avail-gb 6 --label chain_bench_$name -- $PY $D/stepB_chain_bench.py --batch 16 --seconds 90 --root $root --label $name --out $D/stepB_chain_bench_bs16_$name.json > $D/stepB_chain_bench_$name.log 2>&1; echo "$name bench rc=$?"
  for split in main holdout; do dir=$C; [[ $split == holdout ]] && dir=$C/holdout
    MUSETALK_UNET_BACKEND=trt_stagewise MUSETALK_UNET_STAGEWISE_BATCH=16 MUSETALK_UNET_STAGEWISE_CACHE_DIR=$root MUSETALK_TRT_FALLBACK=0 \
    $G --min-avail-gb 6 --label gunet_${name}_$split -- $PY scripts/validate_unet_backend.py --backend runtime --capture-dir $dir --padded-batch-size 8 --group-captures 2 --limit 0 --warmup 1 --iters 2 --fail-mae 0.01 --fail-max-abs 0.5 --report-path $D/gunet_${name}_$split.json > $D/gunet_${name}_$split.log 2>&1; echo "$name gunet $split rc=$?"; done; }
$G --min-avail-gb 8 --label build_srcfp16 -- $PY scripts/build_unet_stagewise.py --batch 16 --opt-level 5 --variant srccache --root models/tensorrt_unet_stagewise_sm89_srcfp16 --blocks prefix,down0rest --report $D/build_srcfp16_report.json > $D/build_srcfp16.log 2>&1; echo "srcfp16 build rc=$?"
gate srcfp16
# srcmix = srcfp16 prefix/down0rest + mixed INT8 down3/mid
python3 - <<'PY'
import json,os
M='models'; a=json.load(open(f'{M}/tensorrt_unet_stagewise_sm89_srcfp16/bs16/manifest.json')); b=json.load(open(f'{M}/tensorrt_unet_stagewise_sm89_mixed/bs16/manifest.json'))
dst=f'{M}/tensorrt_unet_stagewise_sm89_srcmix/bs16'; os.makedirs(dst,exist_ok=True)
blocks=dict(a['blocks']); blocks['down3']=b['blocks']['down3']; blocks['mid']=b['blocks']['mid']
srcs={k:(f'{M}/tensorrt_unet_stagewise_sm89_mixed/bs16' if k in ('down3','mid') else f'{M}/tensorrt_unet_stagewise_sm89_srcfp16/bs16') for k in blocks}
for k,v in blocks.items():
    link=f"{dst}/{v['engine_file']}"; tgt=os.path.realpath(f"{srcs[k]}/{v['engine_file']}")
    if os.path.lexists(link): os.remove(link)
    os.symlink(tgt,link)
a.update(blocks=blocks, complete=False, int8_calibration=b.get('int8_calibration')); [a.pop(k,None) for k in ('probe','runtime','missing_blocks')]
json.dump(a,open(f'{dst}/manifest.json','w'),indent=1); print('srcmix manifest ok')
PY
$G --min-avail-gb 8 --label finalize_srcmix -- $PY scripts/build_unet_stagewise.py --batch 16 --opt-level 5 --variant srccache --root models/tensorrt_unet_stagewise_sm89_srcmix --blocks prefix --report $D/build_srcmix_report.json > $D/build_srcmix.log 2>&1; echo "srcmix finalize rc=$?"
gate srcmix
