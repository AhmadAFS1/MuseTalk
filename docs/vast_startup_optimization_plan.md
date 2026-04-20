# Vast Startup Optimization Plan

## Scope

This note captures the current full-stack Vast.ai bootstrap problem for the
MuseTalk TRT-stagewise server and proposes a practical plan to reduce startup
time without adding paid infrastructure.

This is **not** a replacement for the operational boot docs in
`docs/vast_ai_boot.md`. That document explains how to boot the node. This
document explains what is slow in the latest boot profile and how to reduce it.

Assumptions for this plan:

- full-stack setup is required
  - server runtime deps
  - avatar-prep deps
  - avatar-prep model weights
- the current Vast boot flow remains based on `scripts/vast_onstart.sh`
- the current environment family remains CUDA 12.1 / Python 3.10 / TRT-stagewise
- preferred solutions should avoid recurring storage cost

## Current Latest Setup Profile

The latest `scripts/dependency_install_logs` profile shows that the completed
setup path took **42m45s** before the background server-health phase.

Top-level completed setup phases from the latest log:

- `Install required system packages`: `17s`
- `Build TRT-stagewise experiment venv`: `18m47s`
- `Download and validate model weights`: `23m26s`
- `Validate required model files`: `0s`
- `Run avatar-prep import smoke test`: `9s`
- `Run TRT-stagewise server import smoke test`: `6s`
- `Bootstrap/setup phase finished`: `42m45s`

Important note:

- this specific log ends after `vast_server_ctl.sh` spawned the server process
- it does **not** include a final `health check passed` or total end-to-end boot
  duration
- so the numbers below rank the **completed setup portion**, not the full server
  warmup tail

## Ranked Largest Time Sinks

These are the biggest wall-clock costs from the latest completed setup profile.

1. `Download MuseTalk V1.5 weights`: `16m59s`
   - currently the single largest sink
   - dominated by `models/musetalkV15/unet.pth`
   - latest post-setup validation reported this file at `3.2G`

2. `Install PyTorch CUDA 12.1 wheel set`: `8m15s`
   - dominated by large wheel downloads
   - examples from the same log:
     - `torch`: `780.4 MB`
     - `nvidia-cudnn-cu12`: `664.8 MB`
     - `nvidia-cublas-cu12`: `410.6 MB`
     - `triton`: `209.5 MB`
     - `nvidia-cusparse-cu12`: `196.0 MB`
     - `nvidia-nccl-cu12`: `188.7 MB`

3. `Download SyncNet weights`: `4m14s`
   - dominated by `models/syncnet/latentsync_syncnet.pt`
   - latest post-setup validation reported this file at `1.4G`

4. `Install pinned HLS/api_server dependency set`: `2m44s`

5. `Install torch-tensorrt pinned stack`: `2m37s`
   - includes `tensorrt_cu12_libs-10.3.0` at `2037.5 MB`

6. `Install avatar-preparation dependencies`: `2m33s`

7. `Install pinned export + benchmark dependencies`: `2m15s`

The first three items alone account for about **29m28s** of the `42m45s`
captured setup, so the current problem is now mostly about **large static
artifact transfer**, not Python source builds.

## What Changed Since The Older mmcv Problem

The old `mmcv` compile path used to cost roughly `6-10` minutes.

That is no longer the main issue in the latest profile:

- latest avatar-prep dependency phase: `2m33s`
- latest `Install full mmcv`: `1s`

That is the proof point that the general strategy works:

- when a large or expensive dependency becomes a prebuilt artifact close to the
  install path, boot time drops sharply

The next optimization cycle should apply that same idea to the **current**
largest static assets:

- model files
- pinned wheel sets

## Current Script Behaviors That Still Hurt Boot Time

### 1. The setup path still disables pip caching on heavy installs

`scripts/setup_trt_experiment_env.sh` defines `PIP_CACHE_DIR`, but the heavy
install steps still use `--no-cache-dir`, including:

- PyTorch stack
- TensorRT stack
- export/backend deps
- server deps
- avatar-prep prerequisites

Relevant install sections:

- `scripts/setup_trt_experiment_env.sh`
  - `install_pytorch_cuda_step`
  - `install_torch_tensorrt_step`
  - `install_export_dependencies_step`
  - `install_server_dependencies_step`
  - `install_avatar_prep_prerequisites_step`

Implication:

- even if the machine has previously downloaded these wheels, the current flow
  does not reuse them efficiently

### 2. Full-stack weight downloads are still serialized across repos

`download_weights.sh` currently runs the major groups in sequence:

- MuseTalk V1.5
- SD VAE
- Whisper
- DWPose
- SyncNet
- S3FD
- face-parse model
- ResNet18

Implication:

- independent downloads are not overlapping
- the `16m59s` MuseTalk V1.5 pull and `4m14s` SyncNet pull stack on top of each
  other instead of competing in parallel

### 3. Hugging Face Xet-backed repos are falling back to plain HTTP

The latest log repeatedly reports:

- Xet storage is enabled for the repo
- `hf_xet` is not installed
- the downloader falls back to regular HTTP

Implication:

- the current helper install step is too minimal for the current artifact mix
- increasing `HF_MAX_WORKERS` helps only when there are multiple files to fetch
- it does not solve the largest single-file transfers

### 4. The current full-stack boot still pays for avatar-prep weights every cold node

Because full-stack is required, the current path must fetch:

- `dwpose`
- `syncnet`
- `s3fd`

That is valid, but it means the optimization target is not “remove avatar-prep.”
It is “make avatar-prep assets arrive faster and more predictably.”

## Why Storing All Models In The Main Repo Is The Wrong Move

At first glance this looks similar to the `mmcv` wheel solution, but the scale
is very different.

The full-stack model set is roughly:

- `musetalkV15/unet.pth`: `3.2G`
- `syncnet/latentsync_syncnet.pt`: `1.4G`
- `dwpose/dw-ll_ucoco_384.pth`: `389M`
- `sd-vae/diffusion_pytorch_model.bin`: `320M`
- `whisper/pytorch_model.bin`: `145M`
- `face-parse-bisent/79999_iter.pth`: `51M`
- `face-parse-bisent/resnet18-5c106cde.pth`: about `47M`
- `face_detection/s3fd.pth`: `86M`
- plus the duplicated auxiliary S3FD source file

That is roughly **5.7+ GB** of binary artifacts before counting caches or future
revisions.

Putting those directly in the main Git repo would create several problems:

- clone and pull size would explode
- Git history would bloat permanently
- every source checkout would inherit the artifact weight
- large-binary handling in normal Git is poor
- Git LFS introduces quota and operational complexity, which is not a reliable
  “free forever” answer

Conclusion:

- do **not** store the full model set in the main source repo

## Better No-Cost Approach

Use the same basic idea as the `mmcv` optimization, but move the artifact
storage target.

Recommended direction:

- keep the main source repo small
- publish large immutable artifacts outside the main Git history
- fetch them from a predictable public CDN-backed source
- keep local resume + checksum validation

Two practical no-cost patterns:

### Option A: GitHub release assets in this repo or a companion artifact repo

Store versioned artifacts such as:

- `models-full-stack-core-<version>.tar`
- `models-avatar-prep-<version>.tar`
- `wheelhouse-cu121-trt-<version>.tar`
- `checksums.txt`

Pros:

- no paid storage service required up front
- simple public URLs
- compatible with resumable `curl`
- close in spirit to the successful `mmcv` wheel approach

Cons:

- still large uploads and downloads
- main repo releases become artifact-heavy if kept in the same repo
- should likely live in a separate public artifact repo to keep the source repo
  clean

### Option B: Public artifact repo containing only binaries and manifests

Create a separate public repo dedicated to:

- pinned wheels
- pinned full-stack model bundles
- manifest + checksum files

Pros:

- keeps the main repo clean
- makes artifact versioning explicit
- easy to mirror the current pinned environment

Cons:

- still requires maintaining published bundles
- cloning the artifact repo itself is not the right transport; direct release
  asset download is better

Recommendation:

- prefer **release assets or a companion artifact repo**
- do **not** put the full model set directly in the main Git checkout

## Recommended Improvement Plan

### Phase 1: Fix The Current Downloader And Installer

These are low-risk script changes and should happen first.

1. Preserve pip cache by default.
   - stop using `--no-cache-dir` on the large wheel phases
   - keep a stable cache directory under `/workspace` so a rebuilt venv can
     still reuse wheels

2. Install `hf_xet` before Hugging Face downloads.
   - the latest log explicitly shows Xet-backed repos falling back to regular
     HTTP
   - this is a small dependency with potentially large impact on the biggest
     model pulls

3. Parallelize independent model download groups.
   - current script downloads major repos serially
   - full-stack can safely overlap independent downloads with bounded
     concurrency
   - likely grouping:
     - group A: MuseTalk V1.5
     - group B: SyncNet + DWPose
     - group C: SD VAE + Whisper + face-parse + ResNet18 + S3FD

4. Move caches under a persistent workspace path.
   - pip cache
   - Hugging Face cache
   - any downloader temp cache needed for resumable assets

5. Add checksums to all non-HF direct downloads.
   - face-parse
   - ResNet18
   - any future artifact-bundle downloads

Expected result:

- repeat installs should become much faster
- fresh-node model downloads should improve, especially for HF-backed repos
- the model phase should stop being fully serialized

### Phase 2: Create A Versioned Artifact Distribution Path

This is the real first-boot optimization layer.

1. Publish a versioned full-stack model bundle outside the main repo history.
   - server-core bundle:
     - MuseTalk V1.5
     - SD VAE
     - Whisper
     - face-parse
   - avatar-prep bundle:
     - DWPose
     - SyncNet
     - S3FD

2. Publish a versioned wheelhouse bundle for the pinned Python environment.
   - PyTorch CUDA stack
   - TensorRT stack
   - export/backend deps
   - server deps
   - avatar-prep deps that are worth prepacking

3. Add a manifest file in this repo describing:
   - artifact version
   - expected checksums
   - destination paths
   - fallback source URLs

4. Update bootstrap scripts so they:
   - prefer local/workspace artifact bundles if present
   - otherwise download the published bundles
   - only fall back to piecemeal internet installs if the bundles are missing

Expected result:

- first boot stops depending on many slow third-party endpoints
- the boot path becomes closer to “download a few known bundles and unpack”

### Phase 3: Make Boot Modes Explicit

The repo should separate:

- `fresh full-stack bootstrap`
- `repeat full-stack bootstrap with caches`
- `already-provisioned node startup`

Current pain often comes from treating these as the same operation.

Plan:

1. Keep `SETUP_CLEAN=1` only for forced rebuilds.
2. Add a documented “normal redeploy” path that reuses:
   - model bundles
   - pip wheel cache
   - Hugging Face cache
3. Keep full-stack validation, but do not rebuild what is already valid.

## Concrete Recommendation

If only one medium-sized project is taken on next, it should be this:

1. add persistent pip/HF caches
2. add `hf_xet`
3. parallelize model download groups
4. build a **separate public artifact release path** for:
   - full-stack model bundles
   - pinned wheelhouse bundles

That is the closest scalable extension of the successful `mmcv` idea.

The right analogy is:

- `mmcv` proved that a prebuilt artifact can erase a 6-10 minute build cost
- models and wheel sets are now the same class of problem, just larger
- the answer is still “prebuilt, versioned artifacts”
- the difference is that these artifacts should live **outside the main source
  repo**

## Suggested Implementation Order

### Immediate

- remove `--no-cache-dir` from heavy pip phases
- move pip and HF caches to persistent workspace paths
- install `hf_xet` in the download-helper step
- add parallel model download groups

### Next

- create a public artifact release process for full-stack model bundles
- create a public artifact release process for the pinned wheelhouse
- add manifest + checksum validation

### After That

- adjust bootstrap scripts to prefer artifact bundles first
- keep direct internet installs only as a fallback path

## Decision Summary

- full-stack is required, so the plan must optimize avatar-prep rather than
  remove it
- storing the full model set in the main repo is the wrong tradeoff
- the best no-cost direction is a **separate artifact distribution path**
  combined with persistent caches and parallel downloads
- the `mmcv` win should be copied as a pattern, not copied literally into the
  main source tree
