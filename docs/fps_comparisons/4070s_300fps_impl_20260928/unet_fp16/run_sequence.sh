#!/usr/bin/env bash
# GPU sequence for the FP16 UNet items (plan 1.2, 2.2, 2.3a, 2.3b). Every step is its own
# box_guard lease (< 25 min each). Usage: run_sequence.sh <step> [...]; steps below.
set -uo pipefail
R=/workspace/MuseTalk
D=$R/docs/fps_comparisons/4070s_300fps_impl_20260928/unet_fp16
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
G="$R/scripts/box_guard.sh run --wait-min 30"
C=$R/calibration/unet_multi_avatar_20260928
cd "$R"
for step in "$@"; do
  echo "=== step $step $(date -u +%H:%M:%S)"
  case "$step" in
    A1) $G --min-avail-gb 12 --label stepA_gate -- $PY $D/stepA_cudagraph_gate.py --seconds 30 > $D/stepA_cudagraph_gate.log 2>&1 ;;
    A2) $G --min-avail-gb 12 --label bench_default_sha -- $PY scripts/bench_gpu_path.py --seconds 30 --warmup 10 \
          --label default_flags_unset --out $D/bench_gpu_path_default_30s.json > $D/bench_gpu_path_default_30s.log 2>&1 ;;
    A3) MUSETALK_TRT_UNET_CUDAGRAPHS=manual $G --min-avail-gb 12 --label bench_manual_graph -- $PY scripts/bench_gpu_path.py \
          --seconds 120 --warmup 20 --label trt_ts_bs8_manual_graph --out $D/bench_gpu_path_ts_manualgraph_120s.json \
          > $D/bench_gpu_path_ts_manualgraph_120s.log 2>&1 ;;
    B1_16) $G --min-avail-gb 8 --label build_bs16_up1 -- /usr/bin/time -v $PY scripts/build_unet_stagewise.py --batch 16 \
          --opt-level 5 --blocks up1 > $D/build_bs16_up1.log 2>&1 ;;
    B2_16) $G --min-avail-gb 8 --label build_bs16 -- $PY scripts/build_unet_stagewise.py --batch 16 --opt-level 5 \
          --max-minutes 18 --report $D/build_bs16_report.json > $D/build_bs16.log 2>&1 ;;
    B3_16) $G --min-avail-gb 6 --label chain_bench_bs16 -- $PY $D/stepB_chain_bench.py --batch 16 --seconds 90 > $D/stepB_chain_bench_bs16.log 2>&1 ;;
    B2_8) $G --min-avail-gb 8 --label build_bs8 -- /usr/bin/time -v $PY scripts/build_unet_stagewise.py --batch 8 --opt-level 5 \
          --max-minutes 18 --report $D/build_bs8_report.json > $D/build_bs8.log 2>&1 ;;
    B3_8) $G --min-avail-gb 6 --label chain_bench_bs8 -- $PY $D/stepB_chain_bench.py --batch 8 --seconds 90 > $D/stepB_chain_bench_bs8.log 2>&1 ;;
    G16) for split in main holdout; do dir=$C; [[ $split == holdout ]] && dir=$C/holdout
          MUSETALK_UNET_BACKEND=trt_stagewise MUSETALK_UNET_STAGEWISE_BATCH=16 MUSETALK_TRT_FALLBACK=0 \
          $G --min-avail-gb 6 --label gunet_bs16_$split -- $PY scripts/validate_unet_backend.py --backend runtime \
            --capture-dir $dir --padded-batch-size 8 --group-captures 2 --limit 0 --warmup 1 --iters 2 \
            --fail-mae 0.01 --fail-max-abs 0.5 --report-path $D/gunet_stagewise_bs16_$split.json > $D/gunet_stagewise_bs16_$split.log 2>&1
          echo "G16 $split rc=$?"; done ;;
    G8) for split in main holdout; do dir=$C; [[ $split == holdout ]] && dir=$C/holdout
          MUSETALK_UNET_BACKEND=trt_stagewise MUSETALK_UNET_STAGEWISE_BATCH=8 MUSETALK_TRT_FALLBACK=0 \
          $G --min-avail-gb 6 --label gunet_bs8_$split -- $PY scripts/validate_unet_backend.py --backend runtime \
            --capture-dir $dir --padded-batch-size 8 --limit 0 --warmup 1 --iters 2 \
            --fail-mae 0.01 --fail-max-abs 0.5 --report-path $D/gunet_stagewise_bs8_$split.json > $D/gunet_stagewise_bs8_$split.log 2>&1
          echo "G8 $split rc=$?"; done ;;
    C16) MUSETALK_UNET_BACKEND=trt_stagewise MUSETALK_UNET_STAGEWISE_BATCH=16 MUSETALK_TAESD_WARMUP_BATCHES=16 \
          $G --min-avail-gb 6 --label bench_stagewise_bs16 -- $PY scripts/bench_gpu_path.py --batch 16 --seconds 120 --warmup 20 \
          --label stagewise_bs16_compiled_taesd --out $D/bench_gpu_path_stagewise_bs16_120s.json > $D/bench_gpu_path_stagewise_bs16_120s.log 2>&1 ;;
    C8) MUSETALK_UNET_BACKEND=trt_stagewise MUSETALK_UNET_STAGEWISE_BATCH=8 \
          $G --min-avail-gb 6 --label bench_stagewise_bs8 -- $PY scripts/bench_gpu_path.py --batch 8 --seconds 120 --warmup 20 \
          --label stagewise_bs8_compiled_taesd --out $D/bench_gpu_path_stagewise_bs8_120s.json > $D/bench_gpu_path_stagewise_bs8_120s.log 2>&1 ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
  echo "=== step $step rc=$? $(date -u +%H:%M:%S)"
done
