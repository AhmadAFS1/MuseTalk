#!/usr/bin/env bash
# Step 3: measure an engine set in the full-recipe multi-stream harness, the way the published numbers were measured.
#   scripts/repro_400fps/30_benchmark.sh [engine root] [steps...]
#   steps (default: BENCH T SUST PAIR Q V N15):
#     BENCH per-block engine times, interleaved with the published set    published r5 sum: 31.17 ms per bs16 call
#     T     throughput: 6 streams (six avatars), >= 60 s x 2 repeats      published r5: 401.8 / 400.1 fps
#     SUST  sustained: 5 consecutive >= 60 s repeats                     published r5: 404.0 -> 399.96 fps
#     PAIR  like-for-like: pre-change backends ("BEFORE"), then the set, back to back
#                                                                        published: 252.6 / 251.4 vs 401.4 / 400.1
#     Q     quality capture (raw faces) + quality_ab_metrics.py per avatar vs the accepted renders
#     V     video capture (crf 12 refined mp4 + raw arrays) for 40_videos.sh
#     N15   stability: 15 concurrent streams over the six avatars, 10 x ~63 s windows (stability_report.py)
# All records go under $OUT with the tag "repro_<root suffix>", so re-measuring the published sets never overwrites
# the published records. Nothing else heavy should run on the box meanwhile: the GPU is power-capped at 220 W, and
# any co-runner lowers fps.
source "$(dirname "$0")/lib.sh"
shopt -s nullglob
ROOT="${1:-$ENGINE_ROOT}"; shift || true
STEPS="${*:-BENCH T SUST PAIR Q V N15}"
[ -e "$ROOT/bs16/manifest.json" ] || die "no engine set at $ROOT"
TAG=$(tag "$ROOT")
log "engine set $ROOT, records tagged $TAG under $OUT"
ok() { summary "$1" >/dev/null || { log "$2 FAILED: $(summary "$1")"; return 1; }; }
for s in $STEPS; do
  case $s in
    BENCH) guarded "BENCH_$TAG" 8 "$PY" scripts/bench_stagewise_blocks.py --root "$PUBLISHED_R5" --root "$ROOT" --rounds 9 \
             --out "$OUT/BENCH_$TAG.json" || { log "BENCH failed"; continue; }
           python3 -c "import json; d=json.load(open('$OUT/BENCH_$TAG.json'))['sets']; [print('  %-55s %.2f ms  %s' % (r.split('/')[-1], v['sum_ms'], {k: round(x, 2) for k, x in v['block_ms'].items()})) for r, v in d.items()]" ;;
    T)    harness "T_$TAG" "$ROOT" 12 --streams 6 --loops 18 --repeats 2 --min-timed-s 60; ok "T_$TAG" T && log "T: $(summary T_$TAG)" ;;
    SUST) harness "SUST_$TAG" "$ROOT" 12 --streams 6 --loops 18 --repeats 5 --min-timed-s 60; ok "SUST_$TAG" SUST && log "SUST: $(summary SUST_$TAG)" ;;
    PAIR) ensure_runtime_env
          guarded "PAIR_baseline_$TAG" 12 "$PY" scripts/chin_multistream_render.py --backend baseline --streams 6 \
            --out-root "$OUT/chin_multistream" --label "PAIR_baseline_$TAG" --loops 12 --repeats 2 --min-timed-s 60 --compare-accepted
          ok "PAIR_baseline_$TAG" "PAIR BEFORE (needs the BEFORE .ts engine)" && log "PAIR BEFORE: $(summary PAIR_baseline_$TAG)"
          harness "PAIR_$TAG" "$ROOT" 12 --streams 6 --loops 18 --repeats 2 --min-timed-s 60; ok "PAIR_$TAG" PAIR && log "PAIR $TAG: $(summary PAIR_$TAG)" ;;
    Q)    harness "Q_$TAG" "$ROOT" 10 --streams 6 --loops 1 --save-arrays --compare-accepted; ok "Q_$TAG" Q || continue
          faces=("$OUT/chin_multistream/Q_$TAG"/stream*_faces.npz)
          [ ${#faces[@]} -gt 0 ] || { log "Q: no raw faces captured"; continue; }
          for f in "${faces[@]}"; do
            id=$(basename "$f" _faces.npz); id=${id#stream??_}
            "$PY" scripts/quality_ab_metrics.py pair --identity-dir "$ACCEPTED/$id" --a "dir=$ACCEPTED/$id" \
              --b "faces=$f,label=$TAG" --profile e1 --name "${id}__$TAG" --out-dir "$QRUNS_OUT" > "$OUT/quality_${id}__$TAG.log" 2>&1
            log "quality $id: $(grep -h -o 'VERDICT [A-Z]*' "$OUT/quality_${id}__$TAG.log" | tail -1) (the strict landmark gate fails even on 1-LSB noise; read the table)"
          done
          python3 docs/fps_comparisons/4070s_400fps_20260928/compare_quality.py --runs-dir "$QRUNS_OUT" srcmix_taesdtrt srcg50 "$TAG" | tail -5 ;;
    V)    harness "V_$TAG" "$ROOT" 10 --streams 6 --loops 1 --encode --save-arrays; ok "V_$TAG" V && log "V: $(summary V_$TAG)" ;;
    N15)  harness "N15_$TAG" "$ROOT" 14 --streams 15 --loops 7 --repeats 10 --min-timed-s 60; ok "N15_$TAG" N15 || continue
          log "N15: $(summary N15_$TAG)"
          python3 docs/fps_comparisons/4070s_400fps_20260928/stability_report.py "$OUT/chin_multistream/N15_$TAG.json" \
            --md "$OUT/stability_N15_$TAG.md" | tail -12 ;;
    *)    die "unknown step $s" ;;
  esac
done
