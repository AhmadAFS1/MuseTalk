# Vast Cold-Boot Optimization Plan

## Scope

This document focuses on **cold boot time only** for a brand-new Vast.ai node.

Cold boot means:

- new server
- no existing venv
- no `/workspace` cache
- no existing wheelhouse
- no existing model files

This document intentionally does **not** optimize for:

- repeat boots on the same node
- reusing an existing validated setup
- skipping bootstrap because the environment is already present

Those are useful operational concerns, but they are not the problem being solved
here.

Assumptions:

- full-stack setup is required
  - server runtime deps
  - avatar-prep deps
  - full model set
- current boot path remains based on `scripts/vast_onstart.sh`
- current environment family remains CUDA 12.1 / Python 3.10 / TRT-stagewise
- preferred solutions should avoid recurring cost

## Current Cold-Boot Baseline

The latest `scripts/dependency_install_logs` capture a completed bootstrap/setup
phase of **42m45s** before the server-health phase.

Top-level completed setup phases:

- `Install required system packages`: `17s`
- `Build TRT-stagewise experiment venv`: `18m47s`
- `Download and validate model weights`: `23m26s`
- `Validate required model files`: `0s`
- `Run avatar-prep import smoke test`: `9s`
- `Run TRT-stagewise server import smoke test`: `6s`
- `Bootstrap/setup phase finished`: `42m45s`

Important note:

- this log stops after the server process is spawned
- it does **not** include the final health-ready tail
- so the numbers below rank the **measured cold-bootstrap portion**

## Largest Cold-Boot Time Sinks

These are the biggest wall-clock costs from the latest completed cold bootstrap.

1. `Download MuseTalk V1.5 weights`: `16m59s`
   - dominated by `models/musetalkV15/unet.pth`
   - latest validation reported `unet.pth` at `3.2G`

2. `Install PyTorch CUDA 12.1 wheel set`: `8m15s`
   - dominated by large wheel downloads
   - examples:
     - `torch`: `780.4 MB`
     - `nvidia-cudnn-cu12`: `664.8 MB`
     - `nvidia-cublas-cu12`: `410.6 MB`
     - `triton`: `209.5 MB`
     - `nvidia-cusparse-cu12`: `196.0 MB`
     - `nvidia-nccl-cu12`: `188.7 MB`

3. `Download SyncNet weights`: `4m14s`
   - dominated by `models/syncnet/latentsync_syncnet.pt`
   - latest validation reported `syncnet` at `1.4G`

4. `Install pinned HLS/api_server dependency set`: `2m44s`

5. `Install torch-tensorrt pinned stack`: `2m37s`
   - includes `tensorrt_cu12_libs-10.3.0` at `2037.5 MB`

6. `Install avatar-preparation dependencies`: `2m33s`

7. `Install pinned export + benchmark dependencies`: `2m15s`

The first three items alone account for about **29m28s** of the measured
`42m45s` bootstrap.

Cold-boot conclusion:

- the dominant problem is **multi-GB artifact transfer**
- not Python build logic
- not apt
- not validation steps

## What No Longer Matters As Much

The old `mmcv` source-build problem used to cost roughly `6-10` minutes.

That is no longer the primary cold-boot bottleneck:

- latest avatar-prep dependency phase: `2m33s`
- latest `Install full mmcv`: `1s`

That matters because it proves the pattern:

- when an expensive dependency becomes a nearby prebuilt artifact, cold boot
  drops sharply

The next cold-boot optimization cycle should apply that same pattern to:

- `musetalkV15/unet.pth`
- `syncnet/latentsync_syncnet.pt`
- the PyTorch CUDA wheel set
- the TensorRT wheel set

## What Helps Cold Boot And What Does Not

### Helps Cold Boot

- faster transport for large HF-hosted files
- parallel model-group downloads
- prebuilt artifact bundles for models
- prebuilt artifact bundles for wheel sets
- wheelhouse-first installs when the wheelhouse is prepopulated

### Does Not Materially Solve Cold Boot

- pip cache reuse alone
- skipping bootstrap because the setup already exists
- turning `SETUP_CLEAN` off on a completely new node

Those are repeat-boot optimizations, not cold-boot answers.

## Current Script-Level Cold-Boot Improvements Already Implemented

The repo already has several changes that help cold boots directly:

1. `hf_xet` is now installed during the model download helper step.
2. `HF_XET_HIGH_PERFORMANCE=1` is enabled by default.
3. independent model download groups now run in parallel.
4. heavy dependency families can now prefetch into a wheelhouse in parallel.
5. heavy install phases prefer local wheelhouse artifacts when available.

Cold-boot interpretation:

- these changes can reduce a fresh-node boot
- but they do **not** remove the need to fetch the largest artifacts at least
  once

So they are worthwhile, but they are not sufficient to make a brand-new node
fast by themselves.

## Cold-Boot Priority Targets

If the goal is to materially reduce first boot on a brand-new server, the
highest-value targets are:

### Target 1: MuseTalk V1.5 UNet

- file: `models/musetalkV15/unet.pth`
- latest measured cost: effectively the full `16m59s` MuseTalk V1.5 phase
- latest measured size: `3.2G`

This is the single largest cold-boot target.

### Target 2: SyncNet

- file: `models/syncnet/latentsync_syncnet.pt`
- latest measured cost: `4m14s`
- latest measured size: `1.4G`

This is the second-best model target.

### Target 3: PyTorch CUDA Wheel Family

- latest measured cost: `8m15s`
- dominated by the torch/CUDA wheel family

This is the single largest non-model cold-boot target.

### Target 4: TensorRT Wheel Family

- latest measured cost: `2m37s`
- includes a `2.0G` TensorRT libs wheel

This is the second-largest non-model artifact family.

## Why The Main Git Repo Is Still The Wrong Place For Models

Even with a cold-boot-only lens, putting the full model set directly into the
normal Git history is still the wrong tradeoff.

The full-stack model set is roughly:

- `musetalkV15/unet.pth`: `3.2G`
- `syncnet/latentsync_syncnet.pt`: `1.4G`
- `dwpose/dw-ll_ucoco_384.pth`: `389M`
- `sd-vae/diffusion_pytorch_model.bin`: `320M`
- `whisper/pytorch_model.bin`: `145M`
- `face-parse-bisent/79999_iter.pth`: `51M`
- `face-parse-bisent/resnet18-5c106cde.pth`: about `47M`
- `face_detection/s3fd.pth`: `86M`

That is roughly **5.7+ GB** before revision growth.

Why this should not live in normal Git history:

- the largest files cannot be committed as normal GitHub repo files
- GitHub blocks normal repo files larger than `100 MiB`
- Git LFS per-file limits still create practical issues for multi-GB files
- repo clone/fetch performance would become permanently worse
- every developer checkout inherits the binary weight forever

Cold-boot takeaway:

- use GitHub as a distribution surface if you want
- do **not** use normal Git history as the storage layer

## Best No-Cost Cold-Boot Strategy

The simplest cold-boot strategy that stays GitHub-based is:

### Use GitHub Releases For Prebuilt Bundles

Preferred bundle shape:

- `models-core-part-01`
- `models-core-part-02`
- `models-avatar-prep`
- `wheelhouse-cu121-trt`
- `checksums.txt`

Why this is the best fit:

- still simple
- still GitHub-based
- no paid storage service required up front
- avoids poisoning normal Git history
- works well with resumable `curl`
- matches the successful `mmcv` idea: use prebuilt artifacts instead of building
  or downloading piece by piece from multiple slow endpoints

If per-file size limits require it:

- split the largest model bundle into parts
- reassemble on the node

## Recommended Cold-Boot Plan

### Phase 1: Keep The Current Script-Level Transport Improvements

These are already the right direction for cold boot:

1. `hf_xet`
2. `HF_XET_HIGH_PERFORMANCE=1`
3. parallel model download groups
4. parallel wheelhouse prefetch

These should remain enabled.

### Phase 2: Publish Cold-Boot Artifact Bundles

This is the real next step if cold boot is all that matters.

1. Publish a GitHub release bundle for the full-stack models.
   - core runtime bundle:
     - MuseTalk V1.5
     - SD VAE
     - Whisper
     - face-parse
   - avatar-prep bundle:
     - DWPose
     - SyncNet
     - S3FD

2. Publish a GitHub release bundle for the pinned wheelhouse.
   - PyTorch CUDA wheels
   - TensorRT wheels
   - export/backend wheels
   - server-runtime wheels
   - avatar-prep prerequisite wheels

3. Add checksums for every published artifact.

4. Update bootstrap to:
   - prefer GitHub release bundles first
   - only fall back to public internet piecemeal downloads if the bundles are
     unavailable

Expected cold-boot effect:

- fewer endpoints
- fewer independent transfers
- less dependence on third-party mirror variance
- much better odds of predictable first boot

### Phase 3: Treat Cold Boot As A Separate Product Path

The cold-boot question should be treated separately from repeat-boot behavior.

That means:

- optimize first boot around prebuilt bundles
- stop assuming pip caches are the primary answer
- stop assuming validation-skip logic solves the real problem

## Concrete Recommendation

If the next optimization cycle is focused only on cold boot, the highest-value
project is:

1. keep the current Xet + parallel-download improvements
2. publish GitHub release bundles for:
   - full-stack model artifacts
   - pinned wheelhouse artifacts
3. teach bootstrap to consume those bundles first

That is the cleanest continuation of the `mmcv` strategy.

## Decision Summary

- cold boot is dominated by large artifact transfer
- the two biggest model targets are:
  - `musetalkV15/unet.pth`
  - `syncnet/latentsync_syncnet.pt`
- the two biggest dependency targets are:
  - PyTorch CUDA wheel family
  - TensorRT wheel family
- repeat-boot logic is not the focus here
- the best no-cost cold-boot direction is:
  - **GitHub Releases for prebuilt model and wheel bundles**
  - not normal Git repo storage
