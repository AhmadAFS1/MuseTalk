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

## Latest Validated Cold-Boot Result

The latest validated fresh-server run is captured in
`scripts/dependency_isntall_logs_april_19`.

That run completed the measured setup/bootstrap portion in **18m53s** before the
server-health tail:

- `Install required system packages`: `20s`
- `Build TRT-stagewise experiment venv`: `15m41s`
- `Download and validate model weights`: `2m38s`
- `Validate required model files`: `0s`
- `Run avatar-prep import smoke test`: `8s`
- `Run TRT-stagewise server import smoke test`: `6s`
- `Bootstrap/setup phase finished`: `18m53s`

Important note:

- this log still stops after the server process is spawned
- it does **not** include the final health-ready tail
- so the numbers below rank the **measured cold-bootstrap portion**

Compared with the earlier `42m45s` cold-bootstrap baseline, the measured setup
portion dropped by about **56%**.

## What Improved And Why

The cold-boot picture changed materially in the latest validated run.

### MuseTalk V1.5 Download

The MuseTalk V1.5 download is no longer a `10-15` minute problem.

- older run: `Download MuseTalk V1.5 weights`: `16m59s`
- latest run: `Download MuseTalk model weights`: `2m32s`

That is about an **85% reduction**.

The main reasons are:

1. `hf_xet` is now installed and actually used.
   - older run: Hugging Face logged a fallback to regular HTTP
   - latest run: `Xet Storage is enabled for this repo. Downloading file from Xet Storage..`

2. `HF_XET_HIGH_PERFORMANCE=1` is enabled by default.

3. model groups now download in parallel.

4. legacy MuseTalk V1 weights remain disabled.

Interpretation:

- yes, the V1.5 speedup is primarily the result of getting onto the Xet path
- parallel group downloads and skipping V1 also help the overall wall-clock time
- but the biggest direct evidence in the logs is the switch from HTTP fallback to
  successful Xet-backed transfer

### Full Model Download Phase

The full weight download phase improved even more sharply:

- older run: `23m26s`
- latest run: `2m38s`

That means model transfer is no longer the dominant cold-boot bottleneck in the
validated latest run.

### Avatar-Prep Dependency Phase

The `mmcv` fix is validated.

- `Install full mmcv`: `1s`
- `Install avatar-preparation dependencies`: `1m40s`

The old `6-10` minute `mmcv` source-build problem is no longer relevant on the
current path.

## Current Cold-Boot Bottleneck

In the latest validated run, the dominant measured setup cost is now the venv
build, specifically the API/runtime dependency bundle:

1. `Build TRT-stagewise experiment venv`: `15m41s`
2. inside that, `Phase 6: Install API server runtime dependencies`: `10m00s`
3. inside that, `Install pinned HLS/api_server dependency set`: `9m54s`

So the current bottleneck is no longer MuseTalk V1.5 weights. It is now the
Python dependency set for the API/runtime stack.

Within that phase, the worst observed package downloads in the latest run were:

- `llvmlite-0.47.0` (`56.3 MB`): about `6m03s`
- `av-16.1.0` (`40.3 MB`): about `1m19s`
- `cryptography-46.0.7` (`4.4 MB`): about `44s`
- `scipy-1.15.3` (`37.7 MB`): about `15s`

This is strong evidence that the main remaining cold-boot drag has shifted from
model transport to slow and variable Python package transport.

## What Did Not Work In That Validated Run

The new wheelhouse prefetch optimization did **not** run successfully in that
latest validated build.

Symptoms from the log:

- `Phase 2: Prefetch wheelhouse artifacts` completed in `0s`
- repeated `env_flag_is_true: command not found`

Impact:

- the build did **not** actually exercise the wheelhouse-prefetch/local-wheel
  path for dependency installs
- so the latest validated run still paid the full network cost for the
  API/runtime dependency bundle

Status:

- this helper bug has already been patched in
  `scripts/setup_trt_experiment_env.sh`
- the fix still needs validation on the next fresh-node cold boot

## Latest Follow-Up Status

After the April 20 validated run, the next fresh-node build exercised the new
wheelhouse-prefetch path more fully and exposed a new set of issues.

### What Went Wrong

1. `Prefetch export/backend wheelhouse` resolved the wrong Torch stack.
   - because the prefetch was unconstrained for transitive dependencies, it
     started downloading newer PyPI Torch/CUDA artifacts such as:
     - `torch-2.11.0`
     - CUDA 13 packages
   - this was not the intended runtime stack, which remains Torch `2.5.1` with
     CUDA `12.1`

2. `Prefetch avatar-prep prerequisite wheelhouse` failed on `chumpy`.
   - the isolated wheel-build subprocess failed with:
     - `ModuleNotFoundError: No module named 'pip'`
   - this happened during wheel prefetch, not during the normal install phase

3. the wrapper continued into model download after the env-build phase failed.
   - that created a secondary failure:
     - `Required command not found: huggingface-cli`
   - this was a cascade, not the primary root cause

### What Was Patched

The scripts were updated immediately after that failed run:

1. export/backend wheelhouse prefetch now uses `--no-deps`
   - this prevents it from resolving a newer Torch/CUDA stack during prefetch

2. `chumpy` is no longer prefetched as a wheel
   - it is still installed later in the normal avatar-prep phase with
     `--no-build-isolation`, which is the path that previously worked

3. wheelhouse-prefetch failures are now non-fatal
   - if one prefetch job fails, the build logs the failure and continues with the
     standard install phases instead of aborting immediately

4. `setup_trt_stagewise_server_env.sh` now properly stops if
   `setup_trt_experiment_env.sh` fails
   - this prevents the wrapper from continuing into `download_weights.sh` with an
     incomplete venv

### Current Status

- these fixes are applied locally
- shell syntax validation passes
- they still need validation on the next true fresh-node cold boot

Interpretation:

- the model-side improvements are already validated
- the dependency-side wheelhouse path is still in active validation
- the current goal of the next run is not to prove model download wins again; it
  is to prove that the wheelhouse path no longer breaks the build and actually
  reduces the API/runtime dependency bottleneck

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

- items `1-3` are now validated by the latest fresh-server run
- items `4-5` were present but did **not** take effect in that run because of the
  `env_flag_is_true` helper bug
- the wheelhouse path still needs a clean fresh-server validation after the patch

So the model-side transport improvements are now proven, while the dependency-side
wheelhouse path is only partially implemented until the next validation run.

## Cold-Boot Priority Targets

If the goal is to materially reduce first boot on a brand-new server, the
highest-value targets are:

### Target 1: API Server Runtime Wheel Family

- latest measured cost: `10m00s`
- worst substep: `Install pinned HLS/api_server dependency set` at `9m54s`
- worst observed packages in the latest run:
  - `llvmlite`
  - `av`
  - `cryptography`
  - `scipy`

This is the current largest validated cold-boot target.

### Target 2: Wheelhouse Prefetch Validation

- the wheelhouse prefetch phase currently shows `0s` in the latest validated run
- that happened because of the missing `env_flag_is_true` helper
- the helper has now been patched, but the fix is not yet validated on a fresh node

This is the next immediate engineering step because it may reduce Target 1
substantially without changing the artifact strategy.

### Target 3: MuseTalk V1.5 UNet

- file: `models/musetalkV15/unet.pth`
- latest measured group cost: `2m32s`
- prior measured cost: `16m59s`
- latest measured size: `3.2G`

This is no longer the main blocker, but it is still the single largest model
artifact and still a strong candidate for prebuilt bundle distribution.

### Target 4: SyncNet

- file: `models/syncnet/latentsync_syncnet.pt`
- latest avatar-prep model-group time: `59s`
- latest measured size: `1.4G`

This remains the second-best model artifact target.

### Target 5: PyTorch CUDA And TensorRT Wheel Families

- PyTorch CUDA latest measured cost: `2m11s`
- TensorRT latest measured cost: `59s`

These are no longer the worst phases on this node, but they remain large binary
artifact families and still belong in any eventual prebuilt wheel bundle.

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

### Phase 1: Validate The Wheelhouse Fix On A Fresh Node

This is the immediate next action:

1. run one more true cold boot after the `env_flag_is_true` patch
2. confirm that `Phase 2: Prefetch wheelhouse artifacts` is no longer `0s`
3. confirm that the API/runtime dependency phase drops meaningfully
4. confirm that the log shows local wheelhouse preference during heavy pip phases

This needs to happen before making a larger artifact-publishing decision on the
dependency side.

### Phase 2: Keep The Current Model-Side Transport Improvements

These are already validated and should remain enabled:

1. `hf_xet`
2. `HF_XET_HIGH_PERFORMANCE=1`
3. parallel model download groups
4. skip legacy MuseTalk V1 weights

### Phase 3: Publish Cold-Boot Artifact Bundles

This remains the larger structural answer if cold boot is all that matters.

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

### Phase 4: Treat Cold Boot As A Separate Product Path

The cold-boot question should be treated separately from repeat-boot behavior.

That means:

- optimize first boot around prebuilt bundles
- stop assuming pip caches are the primary answer
- stop assuming validation-skip logic solves the real problem

## Concrete Recommendation

If the next optimization cycle is focused only on cold boot, the highest-value
project is:

1. validate the patched wheelhouse prefetch path on a true fresh-node boot
2. keep the current Xet + parallel-download improvements
3. publish GitHub release bundles for:
   - full-stack model artifacts
   - pinned wheelhouse artifacts
4. teach bootstrap to consume those bundles first

That is the cleanest continuation of the `mmcv` strategy.

## Decision Summary

- the latest validated cold boot completed measured setup in `18m53s`
- MuseTalk V1.5 speed improved from `16m59s` to `2m32s`, primarily because Xet
  is now actually being used
- the full model-download phase improved from `23m26s` to `2m38s`
- the current largest validated bottleneck is now:
  - `Phase 6: Install API server runtime dependencies` at `10m00s`
- the wheelhouse-prefetch optimization was not validated in that run because of
  the helper bug
- the two biggest model artifact targets are still:
  - `musetalkV15/unet.pth`
  - `syncnet/latentsync_syncnet.pt`
- the current biggest dependency target is:
  - the API/runtime wheel family
- repeat-boot logic is not the focus here
- the best no-cost cold-boot direction is:
  - **GitHub Releases for prebuilt model and wheel bundles**
  - not normal Git repo storage
