# WebRTC batch-16 UNet live test results

Date: 2026-07-03

Avatar: `avatar_5a584c8b-512f-4f70-848c-a3d1efbb988f_1782690149`

Audio: `data/audio/ai-assistant.mpga`

## Goal

Test whether the validated batch-16 UNet TensorRT artifact can be promoted into the live WebRTC serving path on one RTX 3090.

Artifact:

`models/tensorrt_unet_static_bs16_20260703_exported_program/unet_trt.ts`

Offline validation had passed before this live run:

- Mean UNet latency: `44.96 ms`
- Mean throughput: `355.86 frames/s`
- MAE max: `0.00177`
- Max abs max: `0.08105`

## Runtime attempts

### 1. VAE TensorRT 8,16 + UNet TensorRT batch 16

Environment:

```bash
WEBRTC_BATCH_FRAME_CALLBACK=1
MUSETALK_BLEND_FIXED_POINT=1
MUSETALK_BLEND_SHRINK_MASK_BBOX=1
HLS_SCHEDULER_MAX_BATCH=16
HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16
MUSETALK_TRT_UNET_ENABLED=1
MUSETALK_TRT_UNET_PATHS=16:models/tensorrt_unet_static_bs16_20260703_exported_program/unet_trt.ts
```

Result: failed at startup after VAE stagewise warmup and face parser load.

Failure:

```text
Torch-TensorRT safeDeserialize OutOfMemory while loading UNet TRT engine.
Requested allocation was about 29 MB.
```

Decision: not viable on the RTX 3090 with the existing VAE TensorRT 8,16 warmup profile.

### 2. VAE TensorRT batch 8 warmup only + UNet TensorRT batch 16

Environment changed:

```bash
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8
```

Result: server started and logged:

```text
UNet backend active: tensorrt_unet_multi
```

First live WebRTC generation failed before producing generated frames. The VAE stagewise backend attempted a runtime compile/allocation and OOMed:

```text
CUDA out of memory. Tried to allocate 1024.00 MiB.
GPU had about 956 MiB free.
```

Decision: also not viable. It only moves the OOM from startup to first request.

### 3. PyTorch VAE + UNet TensorRT batch 16

Environment changed:

```bash
MUSETALK_TRT_ENABLED=0
MUSETALK_VAE_BACKEND=pytorch
MUSETALK_TRT_UNET_ENABLED=1
MUSETALK_TRT_UNET_PATHS=16:models/tensorrt_unet_static_bs16_20260703_exported_program/unet_trt.ts
```

Result: server started and live generation succeeded. This proves the UNet artifact itself can run in the live WebRTC path, but only after disabling the faster VAE TensorRT path.

## Video evidence

Valid smoke video:

`docs/webrtc_unet_bs16_live_tests/2026-07-03/recent_avatar_unet_bs16_pytorch_vae_webrtc_20fps.mp4`

Metadata:

- Codec: `mpeg4`
- Resolution: `720x1280`
- FPS: `20`
- Duration: `14.15s`
- Frames: `283`

Smoke client stats:

- Frames written: `283`
- Video frames played: `110`
- Dropped frames: `0`
- Strict video stalls: `0`

## Capture-disabled WebRTC load tests

These load tests used the only successful live profile: PyTorch VAE plus batch-16 UNet TensorRT.

| Concurrency | Completed | Avg live-ready | Avg frame interval | Max frame interval | Wall time | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C4 | 4/4 | 4.352s | 0.084s | 0.727s | 36.0s | 11,029 MB |
| C6 | 6/6 | 6.299s | 0.129s | 1.581s | 54.5s | 11,033 MB |
| C8 | 8/8 | 8.190s | 0.175s | 2.614s | 72.3s | 11,061 MB |

Summary JSON:

- `webrtc_c4_20fps_unet_bs16_pytorch_vae.json`
- `webrtc_c6_20fps_unet_bs16_pytorch_vae.json`
- `webrtc_c8_20fps_unet_bs16_pytorch_vae.json`

## Server-side timing

Representative successful scheduler timings for the PyTorch VAE plus UNet TRT profile:

| Run | Avg GPU batch | Avg UNet | Avg VAE | Avg compose |
| --- | ---: | ---: | ---: | ---: |
| Smoke | 0.299s | 0.046s | 0.249s | 0.059s |
| C4 representative | 0.297-0.299s | 0.044-0.046s | 0.249-0.250s | 0.040-0.041s |
| C6 representative | 0.300-0.307s | 0.044-0.046s | 0.250-0.252s | 0.038-0.050s |
| C8 representative | 0.303-0.318s | 0.045-0.046s | 0.252-0.262s | 0.037-0.044s |

The UNet TensorRT artifact is fast in isolation and in live serving, but the PyTorch VAE path dominates the GPU batch once VAE TensorRT is disabled.

## Decision

Do not promote the batch-16 UNet TensorRT artifact for the current RTX 3090 WebRTC serving profile.

Reasons:

- The desirable profile, VAE TensorRT 8,16 plus UNet TensorRT batch 16, cannot start on the RTX 3090 because loading the UNet engine OOMs after VAE engines are resident.
- The VAE TensorRT batch-8-only workaround starts but OOMs on first real generation.
- The only successful live profile requires disabling VAE TensorRT, which makes VAE latency about `0.25s` per GPU batch and hurts end-to-end pacing.
- C6 and C8 complete, but they are not smooth 20fps serving results. C8 in particular averages `175ms` frame intervals with a `2.614s` max interval.

Practical capacity from this live test:

- 20fps: C4 is the highest tolerable result in this profile, and even that has tail spikes.
- C6/C8 can complete, but should not be considered production-quality 20fps.

Recommended next step:

Keep the existing optimized VAE TensorRT serving path as the default. Revisit UNet TensorRT only with a smaller artifact, a lower-memory VAE profile, an engine refit/export that lowers load-time workspace, or a scheduler mode that avoids keeping incompatible VAE and UNet engines resident together.
