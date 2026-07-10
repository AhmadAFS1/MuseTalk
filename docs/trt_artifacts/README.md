# MuseTalk TRT Artifact Bundle

Detailed build, validation, visual-test, and load-test report:

```text
docs/trt_artifacts/split8_int8_artifact_run_2026-07-10.md
```

Use the existing MuseTalk S3 bucket with a dedicated prefix:

```text
s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8-current/
```

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
  --uri s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8-current/musetalk-trt-int8-split8.tar.gz
```

For automatic Vast startup restore, set either:

```bash
MUSETALK_TRT_ARTIFACT_URI=s3://lingua-musetalk-s3-storage/trt-artifacts/rtx3090/split8-int8-current/musetalk-trt-int8-split8.tar.gz
```

or set:

```bash
TRT_ARTIFACT_S3_BUCKET=lingua-musetalk-s3-storage
MUSETALK_TRT_ARTIFACT_KEY=trt-artifacts/rtx3090/split8-int8-current/musetalk-trt-int8-split8.tar.gz
```

Runtime servers need `s3:GetObject` for the object. Builder/upload servers also
need `s3:PutObject` for the `trt-artifacts/*` prefix. If the bucket is encrypted
with SSE-KMS, runtime servers also need `kms:Decrypt`.

If the runtime secret already contains `AVATAR_S3_BUCKET`, the secret bootstrap
will also export `TRT_ARTIFACT_S3_BUCKET` with the same value by default. Set
`TRT_ARTIFACT_S3_USE_AVATAR_BUCKET=0` in the secret to opt out.
