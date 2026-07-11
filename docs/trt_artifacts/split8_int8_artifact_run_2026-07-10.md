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

## S3 Publication - Completed 2026-07-11

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

The bundle was uploaded successfully with the dedicated IAM user
`musetalkTRTartifactUpload`. No credential value was printed or committed.
The downloaded access-key CSV was restricted to mode `0600` before use.

Published object:

```text
s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8-current/musetalk-trt-int8-split8.tar.gz
remote size: 1,810,678,414 bytes
bundle sha256: 851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18
multipart ETag observed after upload: a6dffcf7d352b9117c3f5e1b7e2e5549-216
```

After verification, the same object was copied within S3 to the immutable
production key. S3 preserved its size and SHA-256 metadata:

```text
s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8/sha256-851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18/musetalk-trt-int8-split8.tar.gz
```

The `split8-int8-current` object remains available as a mutable alias, but new
servers use the checksum-addressed object.

The multipart ETag is not a SHA-256 checksum and must not be used as the
artifact identity. The uploader stores the SHA-256 in S3 object metadata, and
startup also pins the expected SHA-256 independently.

Post-upload verification downloaded the object into a clean temporary repo
root, extracted it, and ran strict manifest verification:

```text
remote HEAD: passed
clean S3 restore: passed
archive SHA-256: passed
manifest verification: passed
files verified: 49
temporary restore directory removed after verification: yes
```

### IAM setup used for publication

A dedicated IAM user was created instead of using root credentials. Its policy
is limited to this bucket prefix:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListArtifactPrefix",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::lingua-musetalk-s3-storage",
      "Condition": {
        "StringLike": {
          "s3:prefix": ["trt-artifacts/*"]
        }
      }
    },
    {
      "Sid": "ManageTRTArtifacts",
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::lingua-musetalk-s3-storage/trt-artifacts/*"
    }
  ]
}
```

`s3:HeadObject` was intentionally omitted because it is not a valid IAM
action. S3 metadata reads are authorized by `s3:GetObject`.

AWS only displays an IAM secret access key once, when the access key is
created. The downloaded CSV must remain outside Git, be mode `0600` while in
use, and be deleted after the key is transferred to the approved secret store.

## Startup Restore

Startup restore is wired through `scripts/vast_onstart.sh` and
`scripts/trt_artifact_bundle.py`.

Preferred automatic restore variable:

```bash
MUSETALK_TRT_ARTIFACT_URI=s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8/sha256-851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18/musetalk-trt-int8-split8.tar.gz
MUSETALK_TRT_ARTIFACT_SHA256=851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18
MUSETALK_TRT_ARTIFACT_RESTORE=required
MUSETALK_TRT_ARTIFACT_STRICT=1
```

Equivalent bucket/key variables:

```bash
TRT_ARTIFACT_S3_BUCKET=lingua-musetalk-s3-storage
MUSETALK_TRT_ARTIFACT_KEY=trt-artifacts/rtx3090/split8-int8/sha256-851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18/musetalk-trt-int8-split8.tar.gz
MUSETALK_TRT_ARTIFACT_SHA256=851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18
MUSETALK_TRT_ARTIFACT_RESTORE=required
MUSETALK_TRT_ARTIFACT_STRICT=1
```

Runtime servers need `s3:GetObject` for the object. Builder/upload servers need
`s3:PutObject` for the `trt-artifacts/*` prefix. If SSE-KMS is used, runtime
servers also need `kms:Decrypt`.

## Reuse On Every New MuseTalk Server

The recommended design has three credential and artifact boundaries:

1. The publisher IAM user is used only when a validated bundle is uploaded.
2. The Vast bootstrap credential can only read the MuseTalk runtime secret.
3. The S3 runtime credential inside that secret can read the TRT artifact and
   access the other buckets required by the worker.

No runtime-secret content change is required when the existing secret already
contains `AVATAR_S3_BUCKET=lingua-musetalk-s3-storage`: bootstrap derives the
TRT bucket, while the repo pins the key, SHA-256, required mode, and split8
preference. For a deliberate canary or rollback, these optional secret
overrides are available:

```json
{
  "TRT_ARTIFACT_S3_BUCKET": "lingua-musetalk-s3-storage",
  "MUSETALK_TRT_ARTIFACT_KEY": "trt-artifacts/rtx3090/split8-int8/sha256-851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18/musetalk-trt-int8-split8.tar.gz",
  "MUSETALK_TRT_ARTIFACT_SHA256": "851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18",
  "MUSETALK_TRT_ARTIFACT_RESTORE": "required",
  "MUSETALK_TRT_ARTIFACT_STRICT": "1",
  "MUSETALK_TRT_PROFILE_PREFER": "split8"
}
```

The runtime IAM policy must include:

```json
{
  "Effect": "Allow",
  "Action": "s3:GetObject",
  "Resource": "arn:aws:s3:::lingua-musetalk-s3-storage/trt-artifacts/*"
}
```

The Vast template continues to provide only the secret-reader bootstrap
variables documented in `docs/musetalk_worker_secrets.md`. Do not put the S3
runtime key or the artifact publisher key directly in the startup script.

On boot, `scripts/vast_onstart.sh` performs this sequence:

1. Clone the repository and create/validate the Python environment.
2. Fetch and source the runtime secret.
3. Download the TRT bundle from S3.
4. Verify the whole archive against `MUSETALK_TRT_ARTIFACT_SHA256` and the
   SHA-256 stored in S3 metadata.
5. Safely extract the archive and verify all 49 manifest entries.
6. Select the split8 profile.
7. Start the API server and wait for health.

Any missing object, denied S3 request, archive checksum mismatch, unsafe tar
path, missing required artifact, or file checksum mismatch terminates startup
before the API server launches.

The selector uses the restored `models/tensorrt_unet_static_bs8_20260529`
directory and emits fixed/max/startup/warmup batch 8. The published VAE INT8
cache contains batch-8 engines only, so advertising or warming batch 16 would
cause a cache/build mismatch and is intentionally rejected until a separate
batch-16 artifact passes validation.

Restore mode now defaults to `required` in the Vast wrapper. A deliberately
non-TRT server must opt out explicitly with
`MUSETALK_TRT_ARTIFACT_RESTORE=off`; missing configuration is no longer a
silent fallback.

For a new server, the intended Vast launch remains:

```bash
SETUP_CLEAN=1 \
SETUP_FULL_STACK=1 \
STARTUP_TIMEOUT_SECONDS=1800 \
PROFILE=throughput_record \
PORT=8000 \
bash scripts/vast_onstart.sh
```

After startup, verify:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/stats
```

The bootstrap log should contain `TRT artifact restore complete` before the
profile-selection and server-start phases.

## Publishing A Future Artifact

Do not overwrite the production key until the new artifact has passed numeric,
visual, load, and clean-restore testing. Prefer publishing each future bundle
under a versioned or checksum-addressed key first. Update the runtime secret's
key and SHA-256 together, start one canary server, and only then roll the change
to other servers. Keeping the previous key and checksum provides an immediate
rollback.

TensorRT engines are not universally portable. Reuse this bundle only on the
validated compatibility class: RTX 3090/Ampere-compatible GPU, the documented
CUDA/TensorRT/Torch-TensorRT stack, and the exact batch-8 input shapes. A server
with a different GPU architecture or materially different TensorRT runtime
should use an artifact built and validated for that class.

The engine is not tied to an avatar. Avatar identity, idle-video pixels,
bounding boxes, hair, clothing, background, and audio are runtime inputs rather
than constants embedded in the engine. It should therefore work with different
prepared idle videos that satisfy the same preprocessing and tensor-shape
contract. Visual validation in this run covered one avatar, so two or three
additional representative avatars remain prudent release smoke coverage.

The high backend FPS is primarily the GPU TensorRT result. CPU performance can
affect preprocessing, composition, encoding, WebRTC delivery, and whether the
GPU remains fed, but the measured UNet and VAE timings were GPU-stage timings.
Do not report the 20 FPS encoded playback stream as the backend throughput, or
attribute the roughly 94-97 aggregate backend FPS primarily to the CPU.

## Should We Build Batch 16?

Recommendation: keep batch 8 as the production default. Now that its S3 and
clean-restore path is proven, batch 16 is reasonable only as a separate
experiment with its own artifact family and acceptance tests.

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

## Startup Alignment Audit - 2026-07-11

The complete fresh-server chain was reviewed after S3 publication. Two runtime
misalignments were found and corrected:

1. The profile selector looked for a `20260704` batch-8 directory, while the
   published bundle restores the validated `20260529` directory.
2. The selector advertised and warmed batches `8,16`, while the published VAE
   ONNX/QDQ cache contains only the ten batch-8 ONNX/plan files.

The selector now points to the restored `20260529` engine and emits batch 8 for
UNet paths, VAE warmup, scheduler maximum, fixed buckets, and startup slice.
Profile selection now fails closed instead of continuing with normal defaults.
The launcher help text was corrected and `--validate-only` was added for
non-disruptive dependency/artifact checks.

Validation completed on the RTX 3090 server:

```text
canonical immutable S3 object: present
remote size and SHA-256 metadata: matched
clean canonical S3 restore: passed
archive SHA-256 verification: passed
49-file manifest verification: passed
wrong archive SHA-256 rejection before extraction: passed
existing-secret bucket alias test: passed
profile selector path/batch alignment test: passed
launcher --validate-only: passed
shell syntax checks: passed
Python compilation checks: passed
unit tests: 15 passed
live /health: healthy, GPU available
```

The live process was inspected without exposing secrets and matched the target:

```text
VAE backend: trt_stagewise
VAE precision/frontend: int8_mixed / onnx_qdq
VAE warmup/cache batch: 8
UNet backend: trt
UNet path: 8:models/tensorrt_unet_static_bs8_20260529/unet_trt.ts
scheduler max/fixed/startup: 8 / 8 / 8
TRT fallback: disabled
```

One external prerequisite cannot be proven from repository tests: the IAM user
whose credentials are already stored in the runtime secret must have
`s3:GetObject` on
`arn:aws:s3:::lingua-musetalk-s3-storage/trt-artifacts/*`. The dedicated
publisher credential proves the object exists, but it is intentionally not the
credential used by normal workers.
