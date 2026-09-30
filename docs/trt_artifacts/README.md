# MuseTalk TRT Artifact Bundles

Every bundle is a `scripts/trt_artifact_bundle.py` archive: the manifest and
SHA256SUMS sidecars first, then the payload at repo-relative paths. Restores check
the archive sha256 and then every file's sha256. Bundles are checksum-addressed
(`sha256-<archive sha>/` in the key) and never overwritten.

| Bundle | Runs on | Recipe | Restored by |
|---|---|---|---|
| r5 portable (`ampere-plus-r5-srcg50-int8`) | any GPU of compute capability 8.0-9.0 with TensorRT 10.3.0 and >= 8 GB (RTX 3090, 4070 SUPER, 4090, A-series, L40S, A100, H100) | `r5` (default), second candidate | `scripts/vast_onstart.sh` |
| r5 RTX 4070 SUPER (`rtx4070super-r5-srcg50-int8`) | exactly an RTX 4070 SUPER | `r5` (default), first candidate | `scripts/vast_onstart.sh` |
| RTX 3090 split8 (FP16 `.ts` UNet + INT8 SD-VAE) | exactly an RTX 3090 | `legacy_int8` (rollback) | `scripts/vast_onstart.sh` (`MUSETALK_TRT_ARTIFACT_*` defaults) |
| Repro inputs: UNet calibration corpus, harness avatars, live-test audio | data, any machine | none | `scripts/repro_400fps/05_fetch_inputs.sh` |

## Which r5 bundle a host gets

TensorRT plans are compiled for a GPU architecture: plans built on an RTX 4070 SUPER (sm_89) do not load on an
RTX 3090 (sm_86). `configs/recipes/r5.env` therefore lists candidates in order of preference,
`bundle:rtx4070super-r5-srcg50-int8|ampere-plus-r5-srcg50-int8`, and a host gets the first one it fits
(`scripts/musetalk_host_profile.py bundle-check`; `vast_onstart.sh` restores it, the resolver serves it):

- **RTX 4070 SUPER** -> the GPU-specific bundle: ~400 fps (6-stream full recipe, 401.8 / 400.1).
- **Any other Ampere-or-newer GPU** (RTX 3090 included) -> the portable bundle: the same r5 engines built with
  TensorRT hardware compatibility `AMPERE_PLUS`. On the RTX 4070 SUPER itself it measures 307 fps (-23%: the
  architecture-specific kernels are excluded) with the same accuracy; its speed on a 3090 is not measured yet.
  Record: `docs/fps_comparisons/ampere_plus_r5_20260930/README.md`.
- **Older GPUs** (T4, V100, RTX 20xx) fit neither: the server runs eager UNet + compiled TAESD.

A faster, GPU-specific bundle for another model (e.g. a 3090-native one) is added by building it on that GPU and
putting its descriptor before the portable one in the list (see "Publishing a bundle" below).

## r5 portable bundle (`AMPERE_PLUS`)

```text
S3 URI: s3://lingua-musetalk-s3-storage/trt-artifacts/ampere-plus/r5-srcg50-int8/sha256-07644ff16a170ecbfb4eb3d39ff1150731d5508baff31e0216a5dc1bb371df2e/musetalk-trt-r5-ampere-plus.tar.gz
size: 1,000,524,577 bytes (payload 1,163,760,505 bytes, 17 files)
sha256: 07644ff16a170ecbfb4eb3d39ff1150731d5508baff31e0216a5dc1bb371df2e
```

Contents: `models/tensorrt_unet_stagewise_ampere_plus_r5/bs16/` (11 plans, manifest, probe output) and the TensorRT
TAESD engine `models/taesd/trt/taesd_trt_512bfd629a5e1f4f2e40.*` (decoder, fused post, meta, fp16 probe reference;
served with `MUSETALK_TAESD_TRT_HW_COMPAT=ampere_plus`, which the resolver sets). Built 2026-09-30 on an RTX 4070
SUPER, driver 595.84, torch 2.5.1+cu121, TensorRT 10.3.0. On a GPU model other than the build GPU both loaders
accept a probe within relative L2 0.01 of the recorded one (bit-exact on the build GPU).

## r5 RTX 4070 SUPER bundle

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

## Restoring by hand

A bundle's sidecars never go to the repo root (that pair belongs to the RTX 3090 bundle below). They live in its
descriptor's `sidecar_dir` (`.runtime/trt_artifacts/<name>/`) next to a restore stamp that binds them to the archive
sha256; the resolver's `bundle:` prerequisite reads the stamp and checks every file's size. Boot does this
automatically (`docs/STARTUP.md` §3-4). By hand (the runtime credentials can read `trt-artifacts/*`):

```bash
set -a; . /workspace/.musetalk-runtime.env; set +a
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python
N=ampere-plus-r5-srcg50-int8      # or rtx4070super-r5-srcg50-int8
KEY=$(python3 -c "import json; print(json.load(open('configs/trt_bundles/$N.json'))['s3_key'])")
SHA=$(python3 -c "import json; print(json.load(open('configs/trt_bundles/$N.json'))['sha256'])")
# restore: stages the archive in tmp/ (peak disk about archive + payload), verifies, stamps; a later run skips
$PY scripts/trt_artifact_bundle.py --repo-root . --strict --sidecar-dir .runtime/trt_artifacts/$N \
  restore --uri s3://$TRT_ARTIFACT_S3_BUCKET/$KEY --expected-sha256 $SHA --stage-dir tmp/trt_artifact_stage --skip-if-verified
# adopt: the files are already here (built or copied); verify them against the bundle, no download
$PY scripts/trt_artifact_bundle.py --repo-root . --strict --sidecar-dir .runtime/trt_artifacts/$N \
  adopt --uri s3://$TRT_ARTIFACT_S3_BUCKET/$KEY --expected-sha256 $SHA
$PY scripts/musetalk_host_profile.py bundle-check \
  --bundle 'rtx4070super-r5-srcg50-int8|ampere-plus-r5-srcg50-int8'     # which candidates fit / are restored here
```

## Publishing a bundle

Needs an identity with `s3:PutObject` on `trt-artifacts/*` (the runtime credentials are read-only by design). For
a GPU-specific r5 bundle on a new GPU model (e.g. an RTX 3090), on that GPU:

```bash
scripts/repro_400fps/05_fetch_inputs.sh                  # calibration corpus + harness avatars
scripts/repro_400fps/10_build_engines.sh                 # -> models/tensorrt_unet_stagewise_sm86_r5 + TAESD engine
scripts/repro_400fps/20_gate.sh && scripts/repro_400fps/30_benchmark.sh models/tensorrt_unet_stagewise_sm86_r5 BENCH T
$PY scripts/trt_artifact_bundle.py --repo-root . --strict --sidecar-dir tmp/new_bundle_sidecars \
  create --output tmp/new.tar.gz --profile <name> --keep-symlinks --compresslevel 1 \
  --required-files <the TAESD engine's .decoder.plan,.post_bgr_u8.plan,.json> \
  --required-dirs models/tensorrt_unet_stagewise_sm86_r5/bs16
sha256sum tmp/new.tar.gz        # then upload to trt-artifacts/<gpu>/<profile>/sha256-<sha>/<file>.tar.gz
$PY scripts/trt_artifact_bundle.py upload --bundle tmp/new.tar.gz --s3-uri s3://<bucket>/<that key>
```

Then add `configs/trt_bundles/<name>.json` (copy `rtx4070super-r5-srcg50-int8.json`: key, sha256, size,
`host.engine_key` from `scripts/musetalk_engine_keys.py key --kind unet_stagewise`, engine dirs, TAESD key, sidecar
dir) and put `<name>` before the portable candidate in `configs/recipes/r5.env`. The portable bundle itself is
rebuilt with `10_build_engines.sh --hardware-compat ampere_plus`.

## Repro inputs (not needed to serve)

```text
s3://lingua-musetalk-s3-storage/trt-artifacts/repro-inputs/unet-multi-avatar-calibration-20260928/sha256-5b38ed6d0d776b43d405d85839cabeeaf143972ea67c56dd4eaac7e46bab3ded/musetalk-repro-calibration-unet-multi-avatar-20260928.tar.gz
  198,489,230 bytes, 449 files, repo-relative (calibration/unet_multi_avatar_20260928)
s3://lingua-musetalk-s3-storage/trt-artifacts/repro-inputs/avatar-diversity-20260927/sha256-4fbb421484b119814c52ed40840ae41c0481086a53960847b9e215e01b015149/musetalk-repro-avatar-diversity-20260927.tar.gz
  779,866,156 bytes, 300 files, relative to /workspace/experiments (the six harness avatars; not reproducible)
s3://lingua-musetalk-s3-storage/trt-artifacts/repro-inputs/audio-corpus-throughput300/sha256-e4a62439bde6e30e2b25ce5771a8b7f1d6bf0cfd61405f3a498ffadb0550f445/musetalk-repro-audio-corpus-throughput300.tar.gz
  18,658,586 bytes, 39 files, repo-relative (experiments/throughput300_candidate/audio_corpus: the live load test's turns)
```

`scripts/repro_400fps/05_fetch_inputs.sh [--engines]` restores or adopts all three (and, with `--engines`, the r5
bundle that fits this host).

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
