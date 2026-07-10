# Split8 INT8/TRT Artifact Run - 2026-07-10

This document records the batch-8 split8 artifact build, validation, visual
smoke test, and load/performance result from the RTX 3090-compatible MuseTalk
runtime work on 2026-07-10.

## Summary

We rebuilt and validated the runtime profile that produced the high-throughput
result: VAE safe-five INT8 ONNX/QDQ plus a static batch-8 FP16 TensorRT UNet.

Important naming note: this is not a UNet INT8 artifact. INT8 applies to the
VAE decoder stages. The UNet artifact is FP16 TensorRT.

Final target runtime:

```text
VAE backend: TRT stagewise safe-five INT8 ONNX/QDQ
UNet backend: FP16 TensorRT static batch-8
Scheduler: fixed batch 8 / max batch 8
Fallback: disabled for artifact validation
```

## Built Artifacts

UNet TensorRT artifact:

```text
models/tensorrt_unet_static_bs8_20260529/unet_trt.ts
models/tensorrt_unet_static_bs8_20260529/unet_trt_meta.json
```

VAE INT8 artifact inputs/cache:

```text
calibration/vae_decoder/
models/tensorrt/stagewise_int8_onnx_qdq_cache/
```

Captured UNet batch-8 calibration set used for this build:

```text
calibration/unet_static_bs8_20260710/
```

Reusable bundle:

```text
tmp/split8_artifact_20260710/musetalk-trt-int8-split8.tar.gz
sha256: 851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18
```

Bundle sidecars written in the repo root:

```text
.musetalk_trt_artifact_manifest.json
.musetalk_trt_artifact_SHA256SUMS
```

Bundle contents:

```text
files: 49
uncompressed payload bytes: 2472903168
compressed bundle size: about 1.7 GB
UNet artifact size: about 2.1 GB
VAE INT8 cache size: about 236 MB
```

## Runtime Configuration

The target server was launched with the following effective artifact settings:

```bash
MUSETALK_TRT_PROFILE_ENV_LOAD=0
MUSETALK_VAE_BACKEND=trt_stagewise
MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed
MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2
MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR=./calibration/vae_decoder
MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO=minmax
MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR=./models/tensorrt/stagewise_int8_onnx_qdq_cache
MUSETALK_TRT_STAGEWISE_INT8_USE_CACHE=1
MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_CALIBRATION_CACHE=1
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8
HLS_SCHEDULER_MAX_BATCH=8
HLS_SCHEDULER_FIXED_BATCH_SIZES=8
HLS_SCHEDULER_STARTUP_SLICE_SIZE=8
MUSETALK_UNET_BACKEND=trt
MUSETALK_TRT_UNET_ENABLED=1
MUSETALK_TRT_UNET_PATHS=8:models/tensorrt_unet_static_bs8_20260529/unet_trt.ts
MUSETALK_TRT_FALLBACK=0
PROFILE=throughput_record
PORT=8000
```

Startup passed with fallback disabled, proving the required artifacts were
present and loadable.

## Validation Results

UNet TensorRT validation:

```text
report: tmp/split8_artifact_20260710/unet_trt_static_bs8_validation.json
passed: true
precision: float16
padded batch size: 8
files compared: 8
mae mean: 0.0017741761
mae max: 0.0018366629
rmse mean: 0.0030551720
rmse max: 0.0034005661
p95 abs max: 0.005859375
max abs max: 0.098876953125
UNet-only latency mean: 24.182990 ms
UNet-only FPS mean: 330.815860
```

UNet metadata:

```text
type: unet
batch range: 8-8
opt batch: 8
latent shape: [8, 32, 32]
encoder hidden states shape: [50, 384]
dtype: float16
save format: torchscript
validation passed: true
```

VAE safe-five INT8 cache validation/benchmark:

```text
report: tmp/split8_artifact_20260710/vae_safe5_bs8_cache_build.json
label: safe5_bs8_split8_artifact
batch size: 8
int8 stages: decoder_pre, decoder_mid_block, decoder_up_block_0, decoder_up_block_1, decoder_up_block_2
avg decode seconds: 0.0577029244
decode FPS: 138.6411535
MAE vs PyTorch VAE: 0.0050371531
max abs vs PyTorch VAE: 0.128662109375
```

Bundle verification:

```text
Verified TRT artifact manifest: .musetalk_trt_artifact_manifest.json
Files: 49
```

## Visual Smoke Test

The visual smoke test used the documented avatar:

```text
avatar_5a584c8b-512f-4f70-848c-a3d1efbb988f_1782690149
```

Input audio:

```text
data/audio/eng.wav
```

Recorded WebRTC output:

```text
tmp/split8_artifact_20260710/visual_smoke_split8_trt_int8.mp4
```

Stable reference copy:

```text
docs/trt_artifacts/visual_tests/visual_smoke_split8_trt_int8_20260710.mp4
sha256: 6235b51d3e3e626c71465b583aa16497a847becddcac710786332870ae7c2f6b
```

Extracted visual inspection sheets:

```text
tmp/split8_artifact_20260710/visual_frames/contact_sheet_6s.jpg
tmp/split8_artifact_20260710/visual_frames/mouth_motion_10_13s.jpg
```

Visual assessment:

```text
acceptable: yes
identity stability: good
mouth motion: natural variation across speech frames
obvious frozen-mouth artifact: not observed
obvious tearing: not observed
obvious face drift: not observed
```

Recorded video properties:

```text
resolution: 720x1280
duration: 61.9 seconds
frames: 1238
encoded/playback FPS: 20
file size: 13,402,319 bytes
bitrate: 1,732,125 bps
```

WebRTC recording stats:

```text
frames_written: 1238
frames_dropped: 0
frames_duplicated: 0
strict_video_stalls: 0
strict_video_stall_seconds: 0.0
initial_av_start_delta_seconds: about 0.000257
first_audio_packet_after_release_seconds: about 0.07036
first_video_frame_after_release_seconds: about 0.07010
```

The strict FIFO sync clock reported audio wait time during the recording. Video
remained stall-free, and the visual output was acceptable in the extracted
inspection frames.

## Load/Performance Result

Important distinction:

- The WebRTC video stream is encoded and played back at 20 FPS.
- The backend model throughput is measured from GPU batch timings.

Target steady-state GPU batch timings were `actual=8 padded=8`.

Representative final target run timings:

```text
batch size: 8
UNet: about 23-25 ms
VAE: about 58 ms
total batch time: about 82-84 ms
aggregate backend throughput: about 95.7-97.3 FPS
```

Expanded steady-state sample from the target runtime:

```text
samples: 16
mean UNet time: 24.675 ms
mean VAE time: 59.575 ms
mean total batch time: 85.44375 ms
mean aggregate backend throughput: 93.809 FPS
min sampled backend throughput: 80.564 FPS
max sampled backend throughput: 97.324 FPS
```

The earlier headline number of about 96.3 FPS came from the clean final sampled
intervals after startup/warm effects:

```text
95.81 FPS total=0.0835s unet=0.0246s vae=0.0583s
96.15 FPS total=0.0832s unet=0.0236s vae=0.0582s
97.09 FPS total=0.0824s unet=0.0238s vae=0.0578s
95.69 FPS total=0.0836s unet=0.0247s vae=0.0584s
97.32 FPS total=0.0822s unet=0.0234s vae=0.0580s
96.04 FPS total=0.0833s unet=0.0244s vae=0.0584s
```

Including startup/warm and non-target phases in the same log lowers the blended
mean. For reporting this artifact, use the target steady-state range rather
than the mixed-log aggregate.

## S3 Status

The S3 target prefix is:

```text
s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8-current/
```

Upload command:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/trt_artifact_bundle.py \
  upload \
  --bundle tmp/split8_artifact_20260710/musetalk-trt-int8-split8.tar.gz \
  --s3-uri s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8-current/musetalk-trt-int8-split8.tar.gz
```

Upload was not completed from this shell because no AWS credentials were
available to the process:

```text
ERROR: Unable to locate credentials
```

We did not store pasted AWS secret values in the repo or reuse them in commands.

## Startup Restore

Startup restore is wired through `scripts/vast_onstart.sh` and
`scripts/trt_artifact_bundle.py`.

Preferred automatic restore variable:

```bash
MUSETALK_TRT_ARTIFACT_URI=s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8-current/musetalk-trt-int8-split8.tar.gz
```

Equivalent bucket/key variables:

```bash
TRT_ARTIFACT_S3_BUCKET=lingua-musetalk-s3-storage
MUSETALK_TRT_ARTIFACT_KEY=trt-artifacts/rtx3090/split8-int8-current/musetalk-trt-int8-split8.tar.gz
```

Runtime servers need `s3:GetObject` for the object. Builder/upload servers need
`s3:PutObject` for the `trt-artifacts/*` prefix. If SSE-KMS is used, runtime
servers also need `kms:Decrypt`.

## Should We Build Batch 16?

Recommendation: not as the next default target. It is worth testing as a
separate experiment only after the batch-8 artifact is uploaded to S3 and the
new-server restore path is proven.

Reasons:

1. Batch 8 already exceeds the original target. The final target runtime reached
   about 94-97 backend FPS with clean video output.
2. The current runtime bottleneck is mostly VAE time, not UNet time. In the
   target steady state, UNet is about 25 ms while VAE is about 60 ms per batch.
3. Prior notes showed batch-16 VAE safe-five attempts were memory-sensitive and
   could fail with TensorRT OOM/context allocation errors on this stack.
4. A batch-16 UNet artifact would be useful only if the wall regularly runs 16
   active sessions and the VAE/cache side can also sustain that shape without
   memory churn.

If we do try it, treat it as a separate artifact family:

```text
models/tensorrt_unet_static_bs16_YYYYMMDD/unet_trt.ts
models/tensorrt_unet_static_bs16_YYYYMMDD/unet_trt_meta.json
s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/bs16-int8-experimental/
```

Acceptance criteria for batch 16 should be:

```text
UNet validation passed
VAE INT8 cache loads without rebuild
startup succeeds with fallback disabled
16-session wall visual smoke is acceptable
no dropped/duplicated frames in WebRTC recorder
steady-state backend throughput is meaningfully higher than split8
no TensorRT OOM or context allocation failures
```

Until that experiment passes, production/new-server default should remain the
batch-8 split8 artifact.
