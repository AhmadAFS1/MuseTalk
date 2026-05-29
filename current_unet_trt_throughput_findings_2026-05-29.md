# UNet TensorRT Throughput Findings - 2026-05-29

## Decision

Do not enable the current UNet TensorRT artifacts in the live MuseTalk server.

The FP16 TensorRT UNet export can build and serialize through the TorchScript
fallback path, but it fails the captured-output correctness gate. The current
best live runtime remains the validated VAE decoder INT8 `8,16` profile with
PyTorch UNet.

The new Vast.ai host is also capped at `150W` GPU power even though the RTX 3090
default and max limit are `350W`. The container cannot raise it:

```text
nvidia-smi -pl 350
Failed to set power management limit ... Insufficient Permissions
```

Because of that host cap, the clean WebRTC C4/C6/C8 numbers from this run are
useful diagnostics, but they are not comparable to the previous healthier
RTX 3090 throughput runs.

## Environment Sanity

CUDA runtime sanity passed before testing:

| Check | Result |
| --- | --- |
| `torch` | `2.5.1+cu121` |
| `torch.version.cuda` | `12.1` |
| `torch.cuda.is_available()` | `True` |
| `torch.cuda.device_count()` | `1` |
| `libcuda.cuInit(0)` | `0` |
| GPU | `NVIDIA GeForce RTX 3090` |

The server was started with the intended VAE INT8 runtime:

```text
MUSETALK_VAE_BACKEND=trt_stagewise
MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed
MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16
HLS_SCHEDULER_MAX_BATCH=16
HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16
```

Startup proof:

- `VAE decode backend active: tensorrt_stagewise_int8_mixed`
- scheduler `max_combined_batch_size=16`
- avatar `test_avatar` warmed into process cache

## Capture And Validation Corpus

I captured a mixed real WebRTC UNet corpus:

| Batch | Captures |
| ---: | ---: |
| `8` | `16` |
| `16` | `16` |

Directory:

```text
calibration/unet_mixed_8_16_20260529_0318
```

PyTorch reference validation passed on all 32 captures:

| Metric | Value |
| --- | ---: |
| files | `32` |
| `mae_mean` | `0.0002578` |
| `mae_max` | `0.0002788` |
| `rmse_mean` | `0.0005104` |
| `rmse_max` | `0.0006032` |
| `p95_abs_max` | `0.0009766` |
| `max_abs_max` | `0.03418` |

Report:

```text
tmp/unet_pytorch_reference_validation_8_16.json
```

## UNet TensorRT Results

The export script was hardened so a successful compile is not mistaken for a
usable artifact:

- immediate load probe after `torch_tensorrt.save`
- fallback from broken `exported_program` save to TorchScript
- better `scaled_dot_product_attention` matching for `torch_executed_ops`

Validation results:

| Candidate | Artifact | Result |
| --- | --- | --- |
| full FP16 Dynamo TRT | `models/tensorrt/unet_trt.ts` | saved and loadable, failed correctness |
| full FP16 with attention requested on PyTorch, old selector | `models/tensorrt_unet_sdp_pt/unet_trt.ts` | failed correctness; selector did not catch builtin target |
| native group norm requested on PyTorch | `models/tensorrt_unet_gn_pt/unet_trt.ts` | runtime shape failure at mixed batch sizes |
| attention requested on PyTorch, builtin selector fixed | `models/tensorrt_unet_sdp_builtin_pt/unet_trt.ts` | saved and loadable, failed correctness |

Best measured full FP16 TRT candidate:

| Metric | Value |
| --- | ---: |
| passed | `false` |
| files | `32` |
| `mae_mean` | `0.0078095` |
| `mae_max` | `0.0122484` |
| `rmse_mean` | `0.0198268` |
| `rmse_max` | `0.0332933` |
| `p95_abs_max` | `0.0507813` |
| `max_abs_max` | `1.9422` |
| latency mean | `138.41 ms` |
| batch-8 latency mean | `91.69 ms` |
| batch-16 latency mean | `185.13 ms` |
| mean isolated FPS | `86.84` |

Correctness gate failure:

- default gate requires `mae_max <= 0.01`
- default gate requires `max_abs_max <= 0.5`
- this candidate exceeded both, especially max absolute error

The attention-excluded fixed-selector candidate was effectively the same:

| Metric | Value |
| --- | ---: |
| passed | `false` |
| `mae_mean` | `0.0078099` |
| `mae_max` | `0.0122825` |
| `p95_abs_max` | `0.0517205` |
| `max_abs_max` | `1.9537` |

Conclusion: attention alone is not the source of the error. Do not proceed to
UNet INT8 until an FP16 backend passes capture validation.

## Clean WebRTC Baseline On This Host

After disabling capture and leaving UNet on PyTorch, I ran same-host WebRTC
C4/C6/C8 with the VAE INT8 `8,16` profile. The host was power-capped at `150W`.

| Streams | Completed | Avg frame interval | Approx aggregate FPS | Avg live-ready | Max frame interval | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `4/4` | `0.174s` | `23.0` | `10.592s` | `2.478s` | `19040 MB` |
| 6 | `6/6` | `0.260s` | `23.1` | `12.421s` | `4.085s` | `19052 MB` |
| 8 | `8/8` | `0.352s` | `22.7` | `16.585s` | `5.730s` | `19072 MB` |

Reports:

```text
tmp/webrtc_clean_baseline_c4_20260529.json
tmp/webrtc_clean_baseline_c6_20260529_power150w.json
tmp/webrtc_clean_baseline_c8_20260529_power150w.json
```

Server timing on the 150W-capped host was consistently around:

| Component | Current 150W-capped batch-16 turns |
| --- | ---: |
| `avg_gpu_batch` | `0.77-0.78s` |
| `avg_unet` | `0.206-0.210s` |
| `avg_vae` | `0.562-0.570s` |
| `avg_compose` | `0.022-0.024s` |

This explains the flat `~23` aggregate FPS ceiling across C4/C6/C8.

## Comparison To Previous MD Baselines

Previous `docs/musetalk_quantization_optimization_plan.md` 2026-05-27
reference, using five-stage VAE INT8 with `8,16` buckets:

| Streams | Previous `8,16` agg FPS | Current 150W-capped agg FPS | Delta |
| ---: | ---: | ---: | ---: |
| 4 | `64.5` | `23.0` | `-64.3%` |
| 6 | `65.2` | `23.1` | `-64.6%` |
| 8 | `62.5` | `22.7` | `-63.6%` |

Previous `8,16` scheduler timing was roughly:

```text
avg_gpu_batch = 0.20-0.21s
avg_unet      = 0.07-0.08s
avg_vae       = 0.128-0.129s
avg_compose   = 0.064-0.073s
```

Current host timing is about `3.7x` slower on total GPU batch time and about
`4.4x` slower on VAE decode. That is consistent with the `150W` power cap and
should be treated as an infrastructure constraint, not a MuseTalk regression.

Previous strict WebRTC targets still stand:

| Target | Required aggregate FPS | Previous best | Current 150W-capped |
| --- | ---: | ---: | ---: |
| `4 x 20 fps` | `80` | `64.5` | `23.0` |
| `6 x 20 fps` | `120` | `65.2` | `23.1` |
| `8 x 20 fps` | `160` | `62.5` | `22.7` |

## Next Plan

1. Fix the host before drawing more throughput conclusions.
   - Use a Vast offer where RTX 3090 power limit is `350W`, or one where the
     container can set `nvidia-smi -pl 350`.
   - Re-run C4/C6/C8 on the same VAE INT8 `8,16` baseline after the power cap is
     fixed.

2. Keep current UNet TRT artifacts disabled.
   - They are useful diagnostics only.
   - Do not set `MUSETALK_UNET_BACKEND=trt` for these artifacts.

3. Next UNet backend experiment should avoid the current dynamic-shape failure
   modes.
   - Build exact static batch-8 and batch-16 artifacts separately.
   - Dispatch by padded scheduler batch instead of one dynamic `8..16` engine.
   - Re-test group norm and attention kept on PyTorch with static shapes.

4. If full UNet FP16 TRT still fails, switch from monolithic UNet export to
   smaller submodule acceleration.
   - Validate down block, mid block, and up block partitions independently.
   - Keep attention and normalization in PyTorch or FP16 until each partition
     passes capture validation.

5. Only after FP16 UNet passes should INT8 be attempted.
   - Use the real WebRTC capture corpus.
   - Start with conv-heavy subgraphs.
   - Keep mouth-motion visual quality gates stricter than VAE decoder gates.

