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

The latest validated fresh-server comparison is captured in:

- `scripts/rtx_3090_7000mbpsDownload`
- `scripts/rtx_3090ti_700mbpsDownload`

Those two runs completed the measured setup/bootstrap portion in:

- `5m06s` on the RTX 3090 node with roughly `7000 Mbps` download
- `8m58s` on the RTX 3090 Ti node with roughly `700 Mbps` download

For comparison, the earlier April 20 fresh-server baseline in
`scripts/dependency_isntall_logs_april_19` completed the same measured
bootstrap portion in **18m53s**.

Measured top-level timings:

- RTX 3090 / `7000 Mbps`
  - `Install required system packages`: `19s`
  - `Build TRT-stagewise experiment venv`: `4m22s`
  - `Download and validate model weights`: `13s`
  - `Bootstrap/setup phase finished`: `5m06s`
- RTX 3090 Ti / `700 Mbps`
  - `Install required system packages`: `16s`
  - `Build TRT-stagewise experiment venv`: `7m18s`
  - `Download and validate model weights`: `1m12s`
  - `Bootstrap/setup phase finished`: `8m58s`
- April 20 baseline
  - `Install required system packages`: `20s`
  - `Build TRT-stagewise experiment venv`: `15m41s`
  - `Download and validate model weights`: `2m38s`
  - `Bootstrap/setup phase finished`: `18m53s`

Important note:

- these logs still stop after server spawn and short post-setup validation
- they do **not** include the final health-ready tail
- so the numbers below rank the **measured cold-bootstrap portion**

Compared with the April 20 `18m53s` baseline, the measured setup portion
dropped by about:

- **73%** on the `7000 Mbps` node
- **53%** on the `700 Mbps` node

Compared with the much older `42m45s` baseline, the measured setup portion is
down by about:

- **88%** on the `7000 Mbps` node
- **79%** on the `700 Mbps` node

## What Improved And Why

The cold-boot picture changed materially again once the wheelhouse path started
working in the April 22 runs.

### Wheelhouse Prefetch And Local-Wheel Installs

The dependency-side optimization is now validated.

April 20 baseline:

- `Phase 2: Prefetch wheelhouse artifacts`: `0s`
- the helper bug meant the wheelhouse path never actually ran
- `Phase 6: Install API server runtime dependencies`: `10m00s`

April 22 successful runs:

- fast link:
  - `Phase 2: Prefetch wheelhouse artifacts`: `1m08s`
  - `Phase 6: Install API server runtime dependencies`: `27s`
- slower link:
  - `Phase 2: Prefetch wheelhouse artifacts`: `3m57s`
  - `Phase 6: Install API server runtime dependencies`: `23s`

The logs also show direct evidence that installs are now using the prefetched
wheelhouse:

- `Installing ... with local wheelhouse preference`
- `Looking in links: /workspace/.wheelhouse/...`

Interpretation:

- the helper and failure-path fixes are working
- the wheelhouse path no longer breaks the build
- the old API/runtime dependency bottleneck was largely eliminated
- the dependency cost moved earlier into a controlled prefetch phase instead of
  being paid as slow pip resolution/install time later

### MuseTalk V1.5 Download

The MuseTalk V1.5 download is no longer a `10-15` minute problem.

- older run: `Download MuseTalk V1.5 weights`: `16m59s`
- April 20 baseline: `Download MuseTalk model weights`: `2m32s`
- April 22 / `700 Mbps`: `Download MuseTalk model weights`: `50s`
- April 22 / `7000 Mbps`: `Download MuseTalk model weights`: `7s`

That is about:

- **95% faster** than the old run on the `700 Mbps` node
- **99% faster** than the old run on the `7000 Mbps` node

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
- April 20 baseline: `2m38s`
- April 22 / `700 Mbps`: `1m12s`
- April 22 / `7000 Mbps`: `13s`

That means model transfer is no longer the dominant cold-boot bottleneck in the
current validated runs, although link speed still matters materially.

### Avatar-Prep Dependency Phase

The `mmcv` fix is validated.

- April 20 baseline:
  - `Install full mmcv`: `1s`
  - `Install avatar-preparation dependencies`: `1m40s`
- April 22 / `700 Mbps`:
  - `Install full mmcv`: `2s`
  - `Install avatar-preparation dependencies`: `53s`
- April 22 / `7000 Mbps`:
  - `Install full mmcv`: `1s`
  - `Install avatar-preparation dependencies`: `48s`

The old `6-10` minute `mmcv` source-build problem is no longer relevant on the
current path.

## Current Cold-Boot Bottleneck

In the current validated April 22 runs, the dominant measured setup cost is
still the venv build, but the bottleneck inside it has shifted again.

The old bottleneck:

- `Phase 6: Install API server runtime dependencies`: `10m00s` on April 20

The current bottleneck:

- `Phase 2: Prefetch wheelhouse artifacts`

Observed April 22 timings:

- RTX 3090 / `7000 Mbps`
  - `Build TRT-stagewise experiment venv`: `4m22s`
  - `Phase 2: Prefetch wheelhouse artifacts`: `1m08s`
  - `Phase 3: Install core PyTorch CUDA stack`: `59s`
  - `Phase 4: Install TensorRT Python stack`: `39s`
- RTX 3090 Ti / `700 Mbps`
  - `Build TRT-stagewise experiment venv`: `7m18s`
  - `Phase 2: Prefetch wheelhouse artifacts`: `3m57s`
  - `Phase 3: Install core PyTorch CUDA stack`: `59s`
  - `Phase 4: Install TensorRT Python stack`: `46s`

The slowest validated substeps are now:

- RTX 3090 / `7000 Mbps`
  - `Prefetch TensorRT wheelhouse`: `1m05s`
  - `Install PyTorch CUDA 12.1 wheel set`: `59s`
  - `Prefetch PyTorch CUDA wheelhouse`: `44s`
- RTX 3090 Ti / `700 Mbps`
  - `Prefetch TensorRT wheelhouse`: `3m56s`
  - `Prefetch PyTorch CUDA wheelhouse`: `1m02s`
  - `Install PyTorch CUDA 12.1 wheel set`: `59s`

Interpretation:

- API/runtime dependency install is no longer the main blocker
- the remaining setup cost is now dominated by transferring large pinned wheel
  families, especially TensorRT and PyTorch CUDA artifacts
- the large delta between the two April 22 runs is explained much more by
  network throughput than by the GPU model
- for cold boot, GPU choice matters far less than link speed unless the flow is
  doing real compilation or inference work

## What Did Not Work In The April 20 Baseline

The new wheelhouse prefetch optimization did **not** run successfully in that
older validated build.

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
- the fix is now validated by the April 22 successful runs

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
- the next true fresh-node validations now succeeded on both April 22 logs
- the pinned Torch `2.5.1` / CUDA `12.1` path stayed intact
- `chumpy` no longer breaks wheelhouse prefetch
- installs now use the prefetched local wheelhouse as intended

Interpretation:

- the model-side improvements are already validated
- the dependency-side wheelhouse path is now also validated
- the main question is no longer whether the wheelhouse path works
- the new question is how far to push prebuilt bundle distribution to reduce the
  remaining transfer-heavy phases further

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

- items `1-3` are validated
- items `4-5` are now also validated by the April 22 fresh-server runs

So both the model-side and dependency-side cold-boot improvements are now
proven on successful fresh-node runs.

## Cold-Boot Priority Targets

If the goal is to materially reduce first boot on a brand-new server, the
highest-value targets are:

### Target 1: TensorRT Wheel Family

- April 22 / `7000 Mbps`: `Prefetch TensorRT wheelhouse` took `1m05s`
- April 22 / `700 Mbps`: `Prefetch TensorRT wheelhouse` took `3m56s`

This is now the single clearest remaining cold-boot target inside dependency
setup.

### Target 2: PyTorch CUDA Wheel Family

- April 22 / `7000 Mbps`
  - `Prefetch PyTorch CUDA wheelhouse`: `44s`
  - `Install PyTorch CUDA 12.1 wheel set`: `59s`
- April 22 / `700 Mbps`
  - `Prefetch PyTorch CUDA wheelhouse`: `1m02s`
  - `Install PyTorch CUDA 12.1 wheel set`: `59s`

This remains the second-best wheel-family target.

### Target 3: MuseTalk V1.5 UNet

- file: `models/musetalkV15/unet.pth`
- April 22 / `700 Mbps` group cost: `50s`
- April 22 / `7000 Mbps` group cost: `7s`
- older measured cost: `16m59s`

This is no longer a major blocker, but it is still the largest single model
artifact and still a good bundle candidate.

### Target 4: Avatar-Prep Model Group

- April 22 / `700 Mbps`: `1m10s`
- April 22 / `7000 Mbps`: `11s`

This remains the most sensitive model group on slower links.

### Target 5: Predictable First-Boot Distribution

- the latest runs are much faster
- but they are still strongly link-speed dependent

That makes prebuilt release bundles attractive even though the current scripts
are now working correctly.

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

This step is complete.

Validated outcome:

1. `Phase 2: Prefetch wheelhouse artifacts` is no longer `0s`
2. heavy install phases now log local wheelhouse preference
3. the API/runtime dependency phase dropped from `10m00s` to `23-27s`
4. fresh-node builds completed successfully on both April 22 logs

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

1. keep the current wheelhouse-first path and Xet + parallel-download
   improvements
2. target the remaining large binary transfer families first:
   - TensorRT wheelhouse
   - PyTorch CUDA wheelhouse
   - the biggest model bundles
3. publish GitHub release bundles for:
   - full-stack model artifacts
   - pinned wheelhouse artifacts
4. teach bootstrap to consume those bundles first

That is the cleanest continuation of the `mmcv` strategy.

## Decision Summary

- the latest validated cold boots completed measured setup in:
  - `5m06s` on the RTX 3090 / `7000 Mbps` node
  - `8m58s` on the RTX 3090 Ti / `700 Mbps` node
- compared with the April 20 `18m53s` baseline, that is about:
  - `73%` faster on the `7000 Mbps` node
  - `53%` faster on the `700 Mbps` node
- MuseTalk V1.5 speed improved from `16m59s` to:
  - `50s` on the `700 Mbps` node
  - `7s` on the `7000 Mbps` node
- the full model-download phase improved from `23m26s` to:
  - `1m12s` on the `700 Mbps` node
  - `13s` on the `7000 Mbps` node
- the current largest validated bottleneck is now:
  - `Phase 2: Prefetch wheelhouse artifacts`
  - especially the TensorRT and PyTorch CUDA wheel families
- the API/runtime dependency install bottleneck was largely removed:
  - from `10m00s` on April 20
  - to `23-27s` on the April 22 runs
- the wheelhouse-prefetch optimization is now validated
- the two biggest model artifact targets are still:
  - `musetalkV15/unet.pth`
  - `syncnet/latentsync_syncnet.pt`
- the current biggest dependency target is:
  - the TensorRT wheel family, then PyTorch CUDA
- repeat-boot logic is not the focus here
- the best no-cost cold-boot direction is:
  - **GitHub Releases for prebuilt model and wheel bundles**
  - not normal Git repo storage
