# MuseTalk TRT Artifact Bundle

Detailed build, validation, visual-test, and load-test report:

```text
docs/trt_artifacts/split8_int8_artifact_run_2026-07-10.md
```

Use the checksum-addressed object as the production restore target:

```text
s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8/sha256-851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18/musetalk-trt-int8-split8.tar.gz
```

The separately published `split8-int8-current` object is a mutable convenience
alias. Production startup uses the checksum-addressed object above.

The bundle stores runtime artifacts only: the validated batch-8 FP16 TensorRT
UNet split8 engine and metadata, VAE INT8 calibration captures, VAE INT8 cache
files, and manifest/checksum sidecars. It does not store the whole repo.

Current expected paths for the historical 70+ FPS RTX 3090 profile:

```text
models/tensorrt_unet_static_bs8_20260529/unet_trt.ts
models/tensorrt_unet_static_bs8_20260529/unet_trt_meta.json
calibration/vae_decoder/
models/tensorrt/stagewise_int8_onnx_qdq_cache/
```

Important: INT8 applies to the VAE safe-five stages. The UNet artifact in this
profile is FP16 TensorRT, not UNet INT8.

Published bundle:

```text
S3 URI: s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8/sha256-851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18/musetalk-trt-int8-split8.tar.gz
size: 1,810,678,414 bytes
sha256: 851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18
files verified after clean restore: 49
```

Visual smoke-test reference:

```text
docs/trt_artifacts/visual_tests/visual_smoke_split8_trt_int8_20260710.mp4
sha256: 6235b51d3e3e626c71465b583aa16497a847becddcac710786332870ae7c2f6b
```

Create and upload once from a healthy matching GPU server:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/trt_artifact_bundle.py \
  --strict create \
  --output /tmp/musetalk-trt-int8-split8.tar.gz

/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/trt_artifact_bundle.py \
  upload \
  --bundle /tmp/musetalk-trt-int8-split8.tar.gz \
  --s3-uri s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8-current/musetalk-trt-int8-split8.tar.gz
```

Restore manually on a new server:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/trt_artifact_bundle.py \
  --strict restore \
  --uri s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8/sha256-851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18/musetalk-trt-int8-split8.tar.gz \
  --expected-sha256 851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18
```

For automatic Vast startup restore, set either:

```bash
MUSETALK_TRT_ARTIFACT_URI=s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8/sha256-851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18/musetalk-trt-int8-split8.tar.gz
MUSETALK_TRT_ARTIFACT_SHA256=851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18
MUSETALK_TRT_ARTIFACT_RESTORE=required
MUSETALK_TRT_ARTIFACT_STRICT=1
```

or set:

```bash
TRT_ARTIFACT_S3_BUCKET=lingua-musetalk-s3-storage
MUSETALK_TRT_ARTIFACT_KEY=trt-artifacts/rtx3090/split8-int8/sha256-851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18/musetalk-trt-int8-split8.tar.gz
MUSETALK_TRT_ARTIFACT_SHA256=851fc69691e715bebdfdc898272ac2f3854b73975843f681d6ea8236d275be18
MUSETALK_TRT_ARTIFACT_RESTORE=required
MUSETALK_TRT_ARTIFACT_STRICT=1
```

Runtime servers need `s3:GetObject` for the object. Builder/upload servers also
need `s3:PutObject` for the `trt-artifacts/*` prefix. If the bucket is encrypted
with SSE-KMS, runtime servers also need `kms:Decrypt`.

If the runtime secret already contains `AVATAR_S3_BUCKET`, the secret bootstrap
will also export `TRT_ARTIFACT_S3_BUCKET` with the same value by default. Set
`TRT_ARTIFACT_S3_USE_AVATAR_BUCKET=0` in the secret to opt out.

Use separate IAM credentials for publishing and serving. The publisher needs
`s3:PutObject` and `s3:GetObject` on `trt-artifacts/*`. A normal MuseTalk server
only needs `s3:GetObject` on that prefix. `HeadObject` is an S3 API operation,
but `s3:HeadObject` is not an IAM action; `s3:GetObject` authorizes metadata
checks.

Do not commit an AWS access-key CSV or put long-lived keys directly in the
startup script. Store runtime credentials in AWS Secrets Manager and leave only
the minimal secret-reader credential in the Vast template. See
`docs/musetalk_worker_secrets.md`.

`scripts/vast_onstart.sh` defaults restore mode to `required`. A missing URI,
denied download, or checksum failure stops startup. Set
`MUSETALK_TRT_ARTIFACT_RESTORE=off` explicitly only for a server that is not
supposed to run this TRT profile.

The selected runtime is fixed at batch 8 throughout: UNet engine batch 8, VAE
INT8 cache batch 8, scheduler max/fixed/startup batch 8, and VAE warmup batch 8.
Do not add batch 16 to the profile without publishing and validating the
corresponding VAE cache and startup behavior.

Run the launcher checks without starting a second API process:

```bash
bash scripts/run_trt_stagewise_server.sh \
  --profile throughput_record \
  --validate-only
```
