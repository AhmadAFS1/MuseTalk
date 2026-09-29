#!/usr/bin/env bash
# Full-recipe validation of one UNet engine set (same procedure as rounds r2/r3):
#   T  throughput: 6 streams (the six accepted identities), TAESD-TRT + 100% chin + refined seam, >= 60 s x 2
#   Q  raw-face capture vs the accepted renders + scripts/quality_ab_metrics.py (profile e1) per identity
#   V  encoded capture for the labelled A/B videos (compose with scripts/video_ab_round.py afterwards)
# usage: validate_candidate.sh <label> <engine root> <step>...     e.g.  ... blkA8 models/tensorrt_unet_stagewise_sm89_x T Q V
set -uo pipefail
R=/workspace/MuseTalk-perf300; cd $R; D=$R/docs/fps_comparisons/4070s_400fps_20260928; O=$D/chin_multistream; mkdir -p $O
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python; ACC=/workspace/experiments/avatar_diversity_20260927
G="scripts/box_guard.sh run --wait-min 120"
label=$1; root=$2; shift 2
H="$PY scripts/chin_multistream_render.py --backend stagewise16_taesdtrt --flags MUSETALK_UNET_STAGEWISE_CACHE_DIR=$root --streams 6 --out-root $O"
for step in "$@"; do
  case $step in
    T) $G --min-avail-gb 10 --label chinms_T_$label -- $H --label T_$label --loops 18 --repeats 2 --min-timed-s 60 > $O/T_$label.log 2>&1
       echo "T $label rc=$? $(grep -a '^SUMMARY' $O/T_$label.log | tail -1 | grep -o '"median_aggregate_fps": [0-9.]*\|"aggregate_fps_per_repeat": \[[^]]*\]' | tr '\n' ' ')" ;;
    Q) $G --min-avail-gb 8 --label chinms_Q_$label -- $H --label Q_$label --loops 1 --save-arrays --compare-accepted > $O/Q_$label.log 2>&1
       echo "Q $label capture rc=$?"
       for f in $O/Q_$label/stream*_faces.npz; do id=$(basename $f _faces.npz); id=${id#stream??_}
         $PY scripts/quality_ab_metrics.py pair --identity-dir $ACC/$id --a dir=$ACC/$id --b faces=$f,label=$label --profile e1 \
             --name ${id}__$label > $O/Q_$label/quality_$id.log 2>&1
         echo "quality $label $id rc=$? $(grep -h -o 'verdict[^,]*' $O/Q_$label/quality_$id.log | tail -1)"
       done ;;
    V) $G --min-avail-gb 8 --label chinms_V_$label -- $H --label V_$label --loops 1 --encode --save-arrays > $O/V_$label.log 2>&1
       echo "V $label rc=$?" ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
done
