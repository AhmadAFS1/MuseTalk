# MuseTalk TRT Artifact Bundles

Every bundle is a `scripts/trt_artifact_bundle.py` archive: the manifest and
SHA256SUMS sidecars first, then the payload at repo-relative paths. Restores check
the archive sha256 and then every file's sha256. Bundles are checksum-addressed
(`sha256-<archive sha>/` in the key) and never overwritten.

| Bundle | Recipe | Restored by |
|---|---|---|
| RTX 4070 SUPER r5 + r2 (stagewise INT8, TAESD TRT) | `r5` | `scripts/vast_onstart.sh`, from `configs/trt_bundles/rtx4070super-r5-srcg50-int8.json` |
| RTX 3090 split8 (FP16 `.ts` UNet + INT8 SD-VAE) | `legacy_int8` | `scripts/vast_onstart.sh` (`MUSETALK_TRT_ARTIFACT_*` defaults) |
| Repro inputs: UNet calibration corpus, harness avatars | none | `scripts/repro_400fps/05_fetch_inputs.sh` |

## RTX 4070 SUPER r5 bundle (recipe r5)

```text
S3 URI: s3://lingua-musetalk-s3-storage/trt-artifacts/rtx4070super/r5-srcg50-int8/sha256-8e3f4b56dfb9cd82beaca20a22f031fbce4e2bc2a9280f55effb3e73c233ebff/musetalk-trt-r5-r2-rtx4070super.tar.gz
size: 2,579,327,644 bytes (payload 2,864,912,312 bytes, 535 files)
sha256: 8e3f4b56dfb9cd82beaca20a22f031fbce4e2bc2a9280f55effb3e73c233ebff
```

Contents: r5 = `models/tensorrt_unet_stagewise_sm89_srcg50` (layer-selective INT8, `gmac_0.50`, ~400 fps) and
r2 = `..._srcmix` (350 fps), both as relative symlinks into the four block folders that are also included; the
TensorRT TAESD engine `models/taesd/trt/taesd_trt_6111388248264a4ef2ae.*`; the INT8 calibration corpus
`calibration/unet_multi_avatar_20260928`; the load-test audio `experiments/throughput300_candidate/audio_corpus`.
Valid only on NVIDIA GeForce RTX 4070 SUPER, sm_89, TensorRT 10.3.0 (engine key
`sm89-nvidia-geforce-rtx-4070-super-trt10.3.0`), torch 2.5.1+cu121; built on driver 595.84.

Its sidecars never go to the repo root (that pair belongs to the RTX 3090 bundle below). They live in
`.runtime/trt_artifacts/rtx4070super-r5-srcg50-int8/` next to a restore stamp that binds them to the archive
sha256; the resolver's `bundle:` prerequisite reads the stamp and checks every file's size.

Boot (automatic): `MUSETALK_RECIPE=r5 bash scripts/vast_onstart.sh`; see `docs/STARTUP.md` §3-4.

By hand (the runtime credentials can read `trt-artifacts/*`):

```bash
set -a; . /workspace/.musetalk-runtime.env; set +a
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
# restore: stages the archive in tmp/ (about 5.5 GB peak disk), verifies, stamps; a later run skips the download
$PY scripts/trt_artifact_bundle.py --repo-root . --strict \
  --sidecar-dir .runtime/trt_artifacts/rtx4070super-r5-srcg50-int8 \
  restore --uri s3://$TRT_ARTIFACT_S3_BUCKET/trt-artifacts/rtx4070super/r5-srcg50-int8/sha256-8e3f4b56dfb9cd82beaca20a22f031fbce4e2bc2a9280f55effb3e73c233ebff/musetalk-trt-r5-r2-rtx4070super.tar.gz \
  --expected-sha256 8e3f4b56dfb9cd82beaca20a22f031fbce4e2bc2a9280f55effb3e73c233ebff \
  --stage-dir tmp/trt_artifact_stage --skip-if-verified
# adopt: the files are already here (built or copied); verify them against the bundle, no download
$PY scripts/trt_artifact_bundle.py --repo-root . --strict \
  --sidecar-dir .runtime/trt_artifacts/rtx4070super-r5-srcg50-int8 \
  adopt --uri s3://$TRT_ARTIFACT_S3_BUCKET/trt-artifacts/rtx4070super/r5-srcg50-int8/sha256-8e3f4b56dfb9cd82beaca20a22f031fbce4e2bc2a9280f55effb3e73c233ebff/musetalk-trt-r5-r2-rtx4070super.tar.gz \
  --expected-sha256 8e3f4b56dfb9cd82beaca20a22f031fbce4e2bc2a9280f55effb3e73c233ebff
```

Publishing a new bundle (a rebuilt set, another GPU or TensorRT version) needs an identity with `s3:PutObject` on
`trt-artifacts/*` (the runtime credentials are read-only by design):

```bash
$PY scripts/trt_artifact_bundle.py --repo-root . --strict --sidecar-dir tmp/new_bundle_sidecars \
  create --output tmp/new.tar.gz --profile <name> --keep-symlinks --compresslevel 1 \
  --required-files <manifest.json paths> --required-dirs <engine dirs, block dirs, taesd trt files>
sha256sum tmp/new.tar.gz        # then upload to trt-artifacts/<gpu>/<profile>/sha256-<sha>/<file>.tar.gz
$PY scripts/trt_artifact_bundle.py upload --bundle tmp/new.tar.gz --s3-uri s3://<bucket>/<that key>
```

Then add a descriptor next to `configs/trt_bundles/rtx4070super-r5-srcg50-int8.json` (key, sha256, engine key,
engine dirs, sidecar dir) and point a recipe's engine group at it with `requires=bundle:<name>`.

## Repro inputs (not needed to serve)

```text
s3://lingua-musetalk-s3-storage/trt-artifacts/repro-inputs/unet-multi-avatar-calibration-20260928/sha256-5b38ed6d0d776b43d405d85839cabeeaf143972ea67c56dd4eaac7e46bab3ded/musetalk-repro-calibration-unet-multi-avatar-20260928.tar.gz
  198,489,230 bytes, 449 files, repo-relative (calibration/unet_multi_avatar_20260928)
s3://lingua-musetalk-s3-storage/trt-artifacts/repro-inputs/avatar-diversity-20260927/sha256-4fbb421484b119814c52ed40840ae41c0481086a53960847b9e215e01b015149/musetalk-repro-avatar-diversity-20260927.tar.gz
  779,866,156 bytes, 300 files, relative to /workspace/experiments (the six harness avatars; not reproducible)
```

`scripts/repro_400fps/05_fetch_inputs.sh [--engines]` restores or adopts both (and, with `--engines`, the r5 bundle).

## RTX 3090 split8 bundle (recipe legacy_int8)

Detailed build, validation, visual-test, and load-test report:

```text
docs/trt_artifacts/split8_int8_artifact_run_2026-07-10.md
```

For `MUSETALK_RECIPE=legacy_int8` only, the restore target is:

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

The legacy_int8 recipe restores this bundle automatically: `scripts/vast_onstart.sh` pins the key, the sha256
and `required`, and the secret bootstrap derives the bucket, so nothing needs to be set. To override it, set either:

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

This restore runs only when `MUSETALK_RECIPE=legacy_int8`, where it defaults to
`required`: a missing URI, denied download, or checksum failure stops startup.
fast and fast300 never run it; r5 restores the RTX 4070 SUPER bundle instead.
Its sidecars are the tracked `.musetalk_trt_artifact_manifest.json` /
`.musetalk_trt_artifact_SHA256SUMS` in the repo root.

In the legacy split8 profile the runtime is fixed at batch 8 throughout: UNet engine batch 8, VAE
INT8 cache batch 8, scheduler max/fixed/startup batch 8, and VAE warmup batch 8.
Do not add batch 16 to the profile without publishing and validating the
corresponding VAE cache and startup behavior.

Run the launcher checks without starting a second API process:

```bash
bash scripts/run_trt_stagewise_server.sh \
  --profile throughput_record \
  --validate-only
```
