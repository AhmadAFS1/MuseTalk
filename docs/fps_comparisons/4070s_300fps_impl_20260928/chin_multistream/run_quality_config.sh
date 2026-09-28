#!/usr/bin/env bash
# usage: run_quality_config.sh <label> <harness backend> [extra --flags K=V,...]
# 1) chin harness capture (loops 1, raw faces + arrays) through the lease; 2) quality_ab_metrics pair (E1, raw faces) per identity.
set -uo pipefail
R=/workspace/MuseTalk-perf300; cd $R; O=$R/docs/fps_comparisons/4070s_300fps_impl_20260928/chin_multistream
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; ACC=/workspace/experiments/avatar_diversity_20260927
label=$1; backend=$2; flags=${3:-}
scripts/box_guard.sh run --min-avail-gb 8 --wait-min 120 --label chinms_Q_$label -- $PY scripts/chin_multistream_render.py --label Q_$label --backend $backend ${flags:+--flags $flags} --streams 6 --loops 1 --save-arrays --compare-accepted > $O/Q_$label.log 2>&1
echo "capture $label rc=$?"
for f in $O/Q_$label/stream*_faces.npz; do id=$(basename $f _faces.npz); id=${id#stream??_}
  $PY scripts/quality_ab_metrics.py pair --identity-dir $ACC/$id --a dir=$ACC/$id --b faces=$f,label=$label --profile e1 --name ${id}__$label > $O/Q_$label/quality_$id.log 2>&1
  echo "quality $label $id rc=$? $(grep -h -o 'verdict[^,]*' $O/Q_$label/quality_$id.log | tail -1)"
done
