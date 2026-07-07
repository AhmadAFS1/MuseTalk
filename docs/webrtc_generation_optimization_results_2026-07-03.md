# WebRTC Generation Optimization Results, 2026-07-03

This run tested the requested RTX 3090 MuseTalk throughput optimizations against
recent avatar `avatar_5a584c8b-512f-4f70-848c-a3d1efbb988f_1782690149`.

## Executive Summary

The best immediate production change is the WebRTC batch handoff plus compose
ROI shrink/fixed-point blend. The compose microbench improved `compose_frame()`
from `5.89 ms` mean to `1.89 ms` mean, a `3.12x` speedup, with max pixel diff
of `1` over 100 compared frames.

The `8,12` scheduler profile is not better for 20 fps WebRTC C4 on this 3090.
It reduced peak VRAM from `23875 MB` to `20521 MB`, but worsened average frame
interval from `88 ms` to `97 ms`, worsened max interval from `0.864s` to
`0.994s`, and had a slower cold startup.

The new exact batch-16 UNet export is promising. It passed capture validation
and measured `44.96 ms` mean on batch 16, versus `70.07 ms` for the PyTorch
reference validation on the same captures. Do not promote it directly yet:
enable it behind `MUSETALK_TRT_UNET_PATHS=16:...`, run a visual smoke, then run
WebRTC C4/C6/C8 without capture.

The VAE late-block `decoder_up_block_3_resnet_0` batch-16 path is still not
ready. After installing the missing ModelOpt dependencies, the safe-five
batch-16 VAE path OOMed, and the isolated `decoder_up_block_3_resnet_0` ONNX/QDQ
INT8 build failed because TensorRT could not find an implementation after
memory-related tactic skips.

## 2026-07-04 Follow-Up: UNet Batch-8 vs Batch-16

Built and validated both requested UNet TRT artifacts from fresh captures:

| Artifact | Validation | Mean latency | Mean backend fps | Quality gate |
| --- | --- | ---: | ---: | --- |
| `models/tensorrt_unet_static_bs8_20260704/unet_trt.ts` | pass | `25.86 ms` at batch 8 | `309.36 fps` | `mae_max=0.001772`, `max_abs_max=0.1323` |
| `models/tensorrt_unet_static_bs16_20260704/unet_trt.ts` | pass | `44.89 ms` at batch 16 | `356.44 fps` | `mae_max=0.001785`, `max_abs_max=0.1240` |

Validation reports:

- `tmp/unet_trt_bs8_bs16_20260704/unet_trt_static_bs8_validation.json`
- `tmp/unet_trt_bs8_bs16_20260704/unet_trt_static_bs16_validation.json`

Live WebRTC tests used VAE INT8 safe-five
`decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2`,
batch frame handoff, fixed-point/shrunk-mask compose, recent avatar
`avatar_5a584c8b-512f-4f70-848c-a3d1efbb988f_1782690149`, and 20 fps WebRTC.

| Runtime profile | Startup result | C4 avg/max interval | C6 avg/max interval | C8 avg/max interval | Peak VRAM | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Split8 only: `MUSETALK_TRT_UNET_PATHS=8:...`, scheduler `8,16` | healthy in `2m13s` | `0.062s / 0.344s` | `0.093s / 0.901s` | `0.138s / 30.111s` | `20753 MB` | works, but bad C8 tail |
| Dual exact `8,16`: `MUSETALK_TRT_UNET_PATHS=8:...,16:...` | failed startup | n/a | n/a | n/a | OOM after `22723 MB` | do not deploy on 24 GB |
| Exact16 only: `MUSETALK_TRT_UNET_PATHS=16:...`, scheduler `16` | healthy in `1m37s` | `0.061s / 0.219s` | `0.085s / 1.398s` | `0.116s / 2.333s` | `17635 MB` | best validated 3090 profile |

The exact batch-16 artifact is now correctness-valid, but loading both batch-8
and batch-16 UNet engines is too close to the 24 GB RTX 3090 edge. The server
finished VAE INT8 warmup, reached about `22.2 GB`, then failed deserializing the
batch-16 UNet engine with a small CUDA OOM request. The deployable fix is to
run the exact batch-16 engine alone and force the scheduler bucket to `16`.

Video smoke artifact for the selected profile:

- `tmp/unet_trt_bs8_bs16_20260704/video/webrtc_exact16only_20fps_smoke.mp4`
- `frames_written=376`
- WebRTC stats reported `video_stalls=0`, `strict_video_stalls=0`,
  `frames_dropped=0`, `frames_duplicated=0`

Startup has been updated to generate and load the best validated TRT profile:

- `scripts/select_unet_trt_profile.py` writes
  `.runtime/musetalk_trt_best.env`.
- `scripts/vast_onstart.sh` runs the selector after setup/TURN bootstrap and
  before starting the server.
- `scripts/run_trt_stagewise_server.sh` sources the generated env by default.
- Set `MUSETALK_SELECT_BEST_TRT_PROFILE=0` or
  `MUSETALK_TRT_PROFILE_ENV_LOAD=0` to disable this behavior.

Current deployment recommendation: use exact16-only for new 3090 servers. Treat
`6x20fps` as the practical support target from this run. `8x20fps` completes but
is still throttled, so the next concurrency test should be `8x15fps`.

## Code Changes Tested

- Added WebRTC batch frame callback plumbing:
  - `api_server.py`
  - `scripts/hls_gpu_scheduler.py`
  - `scripts/webrtc_tracks.py`
- Added `WEBRTC_BATCH_FRAME_CALLBACK=1` default behavior so composed frame
  batches cross into the asyncio/WebRTC track once per batch instead of once per
  frame.
- Added `push_bgr_frames_batch()` to convert and enqueue a batch inside one
  coroutine.
- Optimized compose in `musetalk/utils/blending.py`:
  - `MUSETALK_BLEND_FIXED_POINT=1` uses integer alpha blending.
  - `MUSETALK_BLEND_SHRINK_MASK_BBOX=1` shrinks the blend ROI to the nonzero
    alpha bounds.
- Added `scripts/benchmark_compose_frame.py` for isolated compose benchmarks.

## Compose Benchmark

Command output:

- `tmp/optimization_experiments_20260703/compose_benchmark_shrink_mask.json`
- `tmp/optimization_experiments_20260703/compose_pixel_diff_baseline_vs_shrunk_fixed.json`

Results:

| Case | Mean | Median | P95 | Max | Single-worker fps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Float full mask | `5.8906 ms` | `5.5477 ms` | `8.0768 ms` | `13.4067 ms` | `169.76` |
| Fixed-point full mask | `6.3033 ms` | `6.1632 ms` | `6.9822 ms` | `10.0044 ms` | `158.65` |
| Fixed-point shrunk mask | `1.8851 ms` | `1.7138 ms` | `2.6000 ms` | `7.2551 ms` | `530.48` |

The fixed-point math alone was not a win. The ROI shrink was the real win, and
the fixed-point shrunk-mask output remained visually safe:

- `frames=100`
- `mae_mean=0.00001258`
- `max_abs=1`
- `pixels_gt1_total=0`

## WebRTC C4 A/B

All tests used:

- `--concurrency 4`
- `--playback-fps 20`
- `--musetalk-fps 20`
- `--batch-size 8`
- `--segment-duration 1`

| Profile | Live-ready avg | Avg interval | Max interval | Wall | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior 8/16 baseline | `4.481s` | `0.091s` | `1.082s` | `38.2s` | `23877 MB` |
| 8/16 + batch handoff + compose opt | `3.914s` | `0.088s` | `0.864s` | `36.6s` | `23875 MB` |
| 8/12 + batch handoff + compose opt | `3.972s` | `0.097s` | `0.994s` | `40.2s` | `20521 MB` |

Artifacts:

- Baseline: `tmp/fps_compare_20260703/load_test_webrtc_recent_avatar_c4_20fps.json`
- 8/16 optimized: `tmp/optimization_experiments_20260703/webrtc_c4_20fps_8_16_batch_handoff_compose_opt_rerun2.json`
- 8/12 optimized: `tmp/optimization_experiments_20260703/webrtc_c4_20fps_8_12_batch_handoff_compose_opt_rerun2.json`

Decision: keep `8,16` for throughput testing. Use `8,12` only as a VRAM-relief
profile when another process needs about `3.3 GB` of headroom.

## UNet Batch-16 Export

Captured 12 real UNet scheduler batches with:

```bash
MUSETALK_UNET_CALIBRATION_CAPTURE=1
MUSETALK_UNET_CALIBRATION_DIR=calibration/unet_bs16_20260703
MUSETALK_UNET_CALIBRATION_MAX_BATCHES=12
HLS_SCHEDULER_MAX_BATCH=16
HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16
```

All 12 captures were `(padded_batch=16, actual_batch=16)`.

PyTorch reference validation:

- Report: `tmp/optimization_experiments_20260703/unet_bs16_pytorch_reference_validation.json`
- Limit: 4 files
- `latency_ms_mean=70.07`
- `frames_per_sec_mean=228.35`
- `mae_max=0.0002496`
- `max_abs_max=0.0205`

TensorRT exact batch-16 export:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/tensorrt_export.py \
  --components unet \
  --batch-sizes 16 \
  --output-dir models/tensorrt_unet_static_bs16_20260703_exported_program \
  --workspace-gb 4 \
  --min-block-size 1 \
  --precision fp16 \
  --save-format exported_program \
  --unet-capture-dir calibration/unet_bs16_20260703 \
  --validate-unet-capture-dir calibration/unet_bs16_20260703 \
  --validate-unet-limit 4 \
  --validate-unet-padded-batch-size 16 \
  --validate-unet-report-path tmp/optimization_experiments_20260703/unet_trt_static_bs16_exported_program_validation_20260703.json \
  --require-valid-unet \
  --torch-executed-op native_group_norm
```

Result:

- Artifact: `models/tensorrt_unet_static_bs16_20260703_exported_program/unet_trt.ts`
- Metadata: `models/tensorrt_unet_static_bs16_20260703_exported_program/unet_trt_meta.json`
- Save note: requested `exported_program`, but TensorRT save failed with
  `NotImplementedError: '__len__' is not implemented for __torch__.torch.classes.tensorrt.Engine`;
  the exporter fell back to `torchscript`.
- Validation report:
  `tmp/optimization_experiments_20260703/unet_trt_static_bs16_exported_program_validation_20260703.json`
- `passed=true`
- `latency_ms_mean=44.96`
- `frames_per_sec_mean=355.86`
- `mae_max=0.001772`
- `max_abs_max=0.08105`

Decision: this is worth the next live-runtime test. It should remove the split8
double-launch overhead for exact batch-16 scheduler turns if runtime memory is
acceptable.

Next gate:

```bash
MUSETALK_TRT_UNET_PATHS=16:models/tensorrt_unet_static_bs16_20260703_exported_program/unet_trt.ts
```

Then run a visual smoke and WebRTC C4/C6/C8 with capture disabled.

## VAE Late-Block Batch-16

The active venv initially lacked ModelOpt dependencies. Installed:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python -m pip install \
  'nvidia-modelopt==0.23.2' 'onnx<1.18' pulp 'nvidia-modelopt[torch]==0.23.2'
```

Safe-five batch-16 benchmark command failed:

```bash
MUSETALK_TRT_STAGEWISE_WORKSPACE_GB=0.25 \
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/benchmark_vae_stagewise_decode.py \
  --label safe5_bs16_workspace025 \
  --batch-size 16 \
  --iters 40 \
  --warmup-iters 5 \
  --calibration-dir calibration/vae_decoder \
  --cache-dir models/tensorrt/stagewise_int8_onnx_qdq_cache \
  --int8-stages decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2 \
  --output-json tmp/optimization_experiments_20260703/vae_safe5_bs16_workspace025_benchmark.json
```

Failure:

- TensorRT OOM during conversion/context setup.
- The earlier default workspace attempt also failed requesting `1073741824`
  bytes during execution context creation.

Isolated `decoder_up_block_3_resnet_0` batch-16 command:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/experiment_vae_decoder_int8.py \
  --stage decoder_up_block_3_resnet_0 \
  --split-up-block 3 \
  --batch-size 16 \
  --calibration-batches 8 \
  --calibration-dir calibration/vae_decoder \
  --cache-dir models/tensorrt/stagewise_int8_up3_resnet0_bs16_cache_20260703 \
  --output-dir tmp/optimization_experiments_20260703/vae_up3_resnet0_bs16_experiment \
  --workspace-gb 0.25 \
  --frontend onnx_qdq \
  --algo minmax \
  --allow-empty-cache
```

Failure report:

- `tmp/optimization_experiments_20260703/vae_up3_resnet0_bs16_experiment/report.json`
- `status=failed`
- Error: `TensorRT failed to build ONNX/QDQ INT8 stage decoder_up_block_3_resnet_0.`
- TensorRT detail: after memory-related tactic skips, it failed with
  `Could not find any implementation for node ... /resnet/conv1/input_quantizer/Cast`.

Decision: do not promote `decoder_up_block_3_resnet_0` batch-16. The next VAE
work should either stay batch-8 for this candidate or move to conv-level
selective INT8 with smaller build surfaces and explicit memory caps.

## Recommendations

1. Keep the WebRTC batch callback enabled by default.
2. Keep compose ROI shrink enabled by default; it is the biggest CPU-side win
   from this run.
3. Keep `8,16` as the main 3090 scheduler profile. Treat `8,12` as a VRAM
   fallback, not a speed profile.
4. Promote the exact batch-16 UNet artifact only after visual smoke plus C4/C6/C8
   WebRTC load tests with capture disabled.
5. Pause whole-stage VAE late-block batch-16. Revisit as a narrower conv-level
   experiment or a batch-8-only experiment.
