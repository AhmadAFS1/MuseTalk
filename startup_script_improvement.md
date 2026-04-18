# Startup Script Improvement Analysis

Last updated: 2026-04-18

## Purpose

This note captures the current startup/setup optimization context for the TRT-stagewise MuseTalk server on Vast.ai.

The deployment goal is not server-only startup. The node must:

- prepare avatars
- create sessions
- host sessions
- generate clips

Because of that, `SETUP_FULL_STACK=1` is the correct target path for this deployment even though it is slower than the server-only path.

## Current Vast.ai Boot Command

```bash
set -euo pipefail

REPO_DIR=/workspace/MuseTalk
REPO_URL="https://github.com/AhmadAFS1/MuseTalk.git"
BRANCH=main

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
  cd "$REPO_DIR"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
fi

cd "$REPO_DIR"

SETUP_CLEAN=1 \
SETUP_FULL_STACK=1 \
STARTUP_TIMEOUT_SECONDS=1800 \
PROFILE=throughput_record \
PORT=8000 \
bash scripts/vast_onstart.sh
```

## Important Runtime Facts

Current `api_server.py` startup uses the MuseTalk v1.5 runtime path:

- `version="v15"`
- `vae_type="sd-vae"`
- `unet_config="./models/musetalkV15/musetalk.json"`
- `unet_model_path="./models/musetalkV15/unet.pth"`
- `whisper_dir="./models/whisper"`

Relevant code: `api_server.py:494-500`

This means:

- the serving/runtime path does **not** need `models/musetalk/pytorch_model.bin`
- the node **does** still need the avatar-prep dependency stack for `/avatars/prepare`
- the full-stack node therefore still needs:
  - `mmengine`
  - `mmcv`
  - `mmdet`
  - `mmpose`
  - `dwpose`
  - `syncnet`
  - `s3fd`

## Instrumentation Already Added

The setup flow is now instrumented with step/phase timing:

- `scripts/lib/step_logging.sh`
- `scripts/setup_trt_experiment_env.sh`
- `scripts/setup_trt_stagewise_server_env.sh`
- `download_weights.sh`

Current behavior:

- every phase and step emits `START` / `DONE`
- elapsed time is printed for each step
- logs are written under `logs/setup`
- the exit summary now prints `Slowest phases` and `Slowest steps`
- `HF_MAX_WORKERS` defaults to `4` unless explicitly overridden

Relevant code:

- `scripts/lib/step_logging.sh:42-56`
- `scripts/lib/step_logging.sh:173-221`
- `download_weights.sh:51-61`
- `scripts/vast_onstart.sh:22-29`
- `scripts/vast_onstart.sh:177-265`

## End-to-End Boot Path

The current cold boot path is:

1. outer Vast command clones or updates the repo
2. `scripts/vast_onstart.sh`
3. `setup_musetalk.sh`
4. `scripts/setup_trt_stagewise_server_env.sh`
5. `scripts/setup_trt_experiment_env.sh`
6. `download_weights.sh`
7. `scripts/vast_server_ctl.sh start`
8. `scripts/run_trt_stagewise_server.sh`
9. `api_server.py` startup
10. `scripts/avatar_manager_parallel.py`
11. `scripts/trt_runtime.py`

The wrapper handoff is explicit in `setup_musetalk.sh:6-9`.

Note:

- the current step-level logging starts inside the repo scripts
- the outer `git clone` / `git fetch` / `git pull` portion of the Vast boot command is not separately broken out in `logs/setup`

## Script-by-Script Analysis

### 1. `scripts/vast_onstart.sh`

Role:

- decides whether setup is needed
- runs setup
- validates the setup
- starts the server and waits for `/health`

Important control points:

- `setup_complete()` checks base runtime files/imports: `scripts/vast_onstart.sh:141-158`
- `avatar_prep_setup_complete()` adds avatar-prep files/imports: `scripts/vast_onstart.sh:160-175`
- `run_setup_if_needed()` handles skip/setup logic: `scripts/vast_onstart.sh:177-266`
- top-level timed phases are in `main()`: `scripts/vast_onstart.sh:341-388`

Important behavior:

- if the existing setup already validates, the script can return early before doing a rebuild
- `SETUP_CLEAN=1` currently influences rebuild arguments only after the existing-setup short-circuit
- this makes repeat benchmarking less intuitive than it looks

Specific issue:

- `run_setup_if_needed()` checks for a valid setup at `scripts/vast_onstart.sh:190-203`
- only later does it honor `SETUP_CLEAN` at `scripts/vast_onstart.sh:221-226`
- result: `SETUP_CLEAN=1` does not by itself force a rebuild if the current setup already passes validation

### 2. `scripts/setup_trt_stagewise_server_env.sh`

Role:

- installs OS packages
- builds the TRT-stagewise venv
- downloads weights
- validates required files
- runs smoke tests

Main steps:

- apt/system packages: `scripts/setup_trt_stagewise_server_env.sh:72-88`
- build venv via experiment-env script: `scripts/setup_trt_stagewise_server_env.sh:90-115`
- download weights: `scripts/setup_trt_stagewise_server_env.sh:117-126`
- validate model files: `scripts/setup_trt_stagewise_server_env.sh:128-155`
- avatar-prep smoke test: `scripts/setup_trt_stagewise_server_env.sh:157-177`
- TRT server smoke test: `scripts/setup_trt_stagewise_server_env.sh:179-198`
- orchestration: `scripts/setup_trt_stagewise_server_env.sh:252-282`

Important behavior:

- exports `DOWNLOAD_MUSETALK_V1_WEIGHTS=0`, so the legacy MuseTalk v1 checkpoint is intentionally skipped for the current server path
- when `--full-stack` is used, it builds one combined venv for serving + avatar prep

### 3. `scripts/setup_trt_experiment_env.sh`

Role:

- this is the real dependency bottleneck
- it creates the Python venv and installs almost the entire runtime and avatar-prep stack

Phase layout:

1. bootstrap venv and packaging tooling
2. install core PyTorch CUDA stack
3. install TensorRT Python stack
4. install export/backend support dependencies
5. install API server runtime dependencies
6. install avatar-preparation dependencies
7. optional broader repo runtime extras

Relevant phase definitions:

- `scripts/setup_trt_experiment_env.sh:505-563`
- `scripts/setup_trt_experiment_env.sh:665-699`

Key observations:

- almost every `pip install` uses `--no-cache-dir`
- that means downloads and built wheels are not reused across rebuilds
- the script exposes `PIP_CACHE_DIR`, but repeated `--no-cache-dir` largely defeats that benefit

Relevant lines:

- `scripts/setup_trt_experiment_env.sh:204-231`
- `scripts/setup_trt_experiment_env.sh:263-277`
- `scripts/setup_trt_experiment_env.sh:421-479`
- `scripts/setup_trt_experiment_env.sh:501-503`

#### Why `mmcv` is slow

The avatar-prep dependency phase is dominated by `mmcv`.

Current resolved environment:

- Python tag: `cp310`
- torch: `2.5.1+cu121`
- CUDA: `12.1`
- requested `mmcv`: `2.1.0`

The script now probes the OpenMMLab index before attempting `mim install`:

- `scripts/setup_trt_experiment_env.sh:339-419`

In the latest run, the probe explicitly reported:

- no OpenMMLab wheel index is published for `cu121/torch2.5.0`
- therefore `mim install mmcv` is skipped
- the script goes straight to source-build fallback

Relevant install path:

- `install_mmcv_step()`: `scripts/setup_trt_experiment_env.sh:437-460`

What happens in practice:

1. `openmim` and prerequisites are installed
2. `mmengine` installs quickly
3. `mmcv` is source-built with `pip install --no-build-isolation`
4. `mmdet` and `mmpose` install afterward

That source build is the single largest dependency hotspot in the latest full-stack run.

#### Avatar-prep prerequisite churn

The prerequisite step installs:

- `openmim`
- `setuptools<81`
- `ninja`
- `psutil`

Relevant code: `scripts/setup_trt_experiment_env.sh:421-427`

In live logs, this step also downgraded a number of packages while installing `openmim`, including:

- `urllib3`
- `tqdm`
- `setuptools`
- `packaging`
- `filelock`
- `requests`

That is not the primary bottleneck, but it does increase dependency churn and makes the avatar-prep phase noisier than necessary.

### 4. `download_weights.sh`

Role:

- downloads all required model assets
- validates required model files

Phase layout:

1. prepare weight download environment
2. download MuseTalk model weights
3. download base runtime model weights
4. download avatar-preparation model weights
5. download face parsing support weights
6. validate downloaded model set

Relevant code:

- helpers and environment: `download_weights.sh:8-61`
- download functions: `download_weights.sh:108-287`
- phase orchestration: `download_weights.sh:293-392`

Important behavior:

- defaults `HF_MAX_WORKERS=4`
- keeps downloads sequential across repos
- `HF_MAX_WORKERS` only helps within one repo download
- `HF_HUB_ENABLE_HF_TRANSFER` is currently forced to `0`
- helper install still runs `pip install ... gdown` each time

Relevant lines:

- `download_weights.sh:21-24`
- `download_weights.sh:51-61`
- `download_weights.sh:108-118`
- `download_weights.sh:229-230`

Implications:

- for one-large-file repos, the per-repo worker count has limited effect
- the script is not currently taking advantage of `hf_xet`
- sequential repo downloads limit parallelism

### 5. `scripts/vast_server_ctl.sh`

Role:

- starts the API server
- writes the PID
- waits until `/health` returns success

Relevant code:

- health polling: `scripts/vast_server_ctl.sh:98-149`
- startup command: `scripts/vast_server_ctl.sh:197-233`

Important behavior:

- `start_server()` does not return until `/health` passes
- this means any startup work inside FastAPI startup blocks the whole Vast on-start timing

### 6. `scripts/run_trt_stagewise_server.sh`

Role:

- exports runtime profile env vars
- launches `api_server.py`

Important defaults:

- `MUSETALK_WARM_RUNTIME=1`
- `MUSETALK_TRT_ENABLED=1`
- `MUSETALK_VAE_BACKEND=trt_stagewise`

Relevant lines:

- `scripts/run_trt_stagewise_server.sh:108-116`

Profile-specific warmup behavior:

- `baseline` warms batch `4`
- `throughput_record` warms batches `8,16`

Relevant lines:

- `scripts/run_trt_stagewise_server.sh:142-158`

This matters because the `throughput_record` profile directly increases cold-start time by forcing the stagewise TRT decoder to warm larger live batch buckets before the server becomes healthy.

### 7. `api_server.py`

Role:

- FastAPI startup initializes the full runtime
- `/health` does not report healthy until startup completes

Relevant startup code:

- `api_server.py:486-579`

Important behavior:

- startup constructs `ParallelAvatarManager(args, max_concurrent_inferences=5)` at `api_server.py:512`
- `/health` returns `503` until that manager exists and is ready

Relevant health code:

- `api_server.py:662-688`

### 8. `scripts/avatar_manager_parallel.py`

Role:

- loads the models
- installs the VAE decode backend
- warms runtime paths

Relevant code:

- `_init_models()`: `scripts/avatar_manager_parallel.py:131-199`
- `_warm_runtime_paths()`: `scripts/avatar_manager_parallel.py:391-413`

Important startup sequence inside `_init_models()`:

1. load VAE, UNet, and PE
2. call `load_vae_trt_decoder(...)`
3. set the VAE decode backend
4. load Whisper and face parsing
5. call `compile_models()`
6. call `_warm_runtime_paths()`

The biggest cold-start impact here is step 2, because `load_vae_trt_decoder()` immediately triggers stagewise TRT warmup.

### 9. `scripts/trt_runtime.py`

Role:

- builds the stagewise TRT VAE decode backend
- warms the exact live batch buckets used by the server profile

Relevant code:

- warmup batch selection: `scripts/trt_runtime.py:184-207`
- batch compilation cache in memory: `scripts/trt_runtime.py:359`
- per-batch ensure/compile: `scripts/trt_runtime.py:423-447`
- stagewise warmup loop: `scripts/trt_runtime.py:450-487`
- backend activation: `scripts/trt_runtime.py:806-849`

Important behavior:

- stagewise engines are cached in `self.compiled_by_batch`
- that cache is in memory, not persisted to disk
- on a fresh process start, the stagewise decoder recompiles its batch buckets again

Important mismatch:

- setup scripts create `ARTIFACT_DIR`, defaulting to `models/tensorrt_altenv_bs32`
- the active `trt_stagewise` runtime path does not currently reuse persisted stagewise engines from that directory during startup
- therefore `ARTIFACT_DIR` is not reducing cold startup for the current stagewise path

## Latest Measured Timing: Full-Stack Setup Run (2026-04-18)

This is the latest clean full-stack setup run captured in `log_thing`.

### A. Top-level `vast_onstart.sh` timing

| Phase | Time |
| --- | ---: |
| Bootstrap/setup phase finished | 14m25s |
| Post-setup validation finished | 5s |
| Server start-to-health phase finished | 0s |
| Overall on-start completed | 14m30s |

Notes:

- `Server start-to-health` was `0s` in this run because the server was already running, so this run is a good dependency/setup measurement but **not** a true cold server-start measurement.
- Evidence in `log_thing`: around `2762-2783`.

### B. `scripts/setup_trt_stagewise_server_env.sh` timing

| Step | Time |
| --- | ---: |
| Install required system packages | 6s |
| Build TRT-stagewise experiment venv | 11m27s |
| Download and validate model weights | 2m39s |
| Validate required model files | 0s |
| Run avatar-prep import smoke test | 8s |
| Run TRT-stagewise server import smoke test | 5s |

Evidence in `log_thing`: around `1515-1547`, `2528-2760`.

### C. `scripts/setup_trt_experiment_env.sh` timing

| Phase | Time |
| --- | ---: |
| Phase 1: Bootstrap venv and packaging tooling | 7s |
| Phase 2: Install core PyTorch CUDA stack | 1m48s |
| Phase 3: Install TensorRT Python stack | 55s |
| Phase 4: Install export and backend support dependencies | 24s |
| Phase 5: Install API server runtime dependencies | 51s |
| Phase 6: Install avatar-preparation dependencies | 7m22s |
| Phase 7: Install broader repo runtime extras | 0s |

Slowest steps from the same run:

| Step | Time |
| --- | ---: |
| Install full mmcv | 6m01s |
| Install PyTorch CUDA 12.1 wheel set | 1m48s |
| Install torch-tensorrt pinned stack | 55s |
| Install pinned HLS/api_server dependency set | 45s |
| Install avatar-preparation prerequisites | 28s |

Evidence in `log_thing`: around `1583-2526`.

### D. `download_weights.sh` timing

| Phase | Time |
| --- | ---: |
| Phase 1: Prepare weight download environment | 4s |
| Phase 2: Download MuseTalk model weights | 1m26s |
| Phase 3: Download base runtime model weights | 10s |
| Phase 4: Download avatar-preparation model weights | 51s |
| Phase 5: Download face parsing support weights | 7s |
| Phase 6: Validate downloaded model set | 0s |

Slowest download steps from the same run:

| Step | Time |
| --- | ---: |
| Download MuseTalk V1.5 weights | 1m26s |
| Download SyncNet weights | 36s |
| Download DWPose weights | 11s |

Evidence in `log_thing`: around `2568-2705`.

## Earlier Fully Cold Run (2026-04-17)

The 2026-04-17 run is still useful because it includes a true cold server start.

### Top-level timing

| Phase | Time |
| --- | ---: |
| Bootstrap/setup phase finished | 26m40s |
| Server start-to-health phase finished | 10m21s |
| Overall on-start completed | 37m06s |

Evidence in `log_thing`: `1461`, `1487`, `1488`.

### Setup timing from that earlier run

| Step | Time |
| --- | ---: |
| Install required system packages | 21s |
| Build TRT-stagewise experiment venv | 21m07s |
| Download and validate model weights | 4m58s |

Evidence in `log_thing`: around `139-1459`.

### Cold startup timing from `api_server_8000.log`

The server log under `/workspace/logs/musetalk/api_server_8000.log` shows the real cold-start bottleneck:

| Startup event | Time |
| --- | ---: |
| Stagewise TRT warmup batches | `[8, 16]` |
| Batch 8 ready | 255.14s |
| Batch 16 ready | 349.35s |
| Total stagewise TRT warmup | 604.50s |

Evidence:

- `/workspace/logs/musetalk/api_server_8000.log:38`
- `/workspace/logs/musetalk/api_server_8000.log:41`
- `/workspace/logs/musetalk/api_server_8000.log:43`
- `/workspace/logs/musetalk/api_server_8000.log:44`

This aligns with the `10m21s` health wait in the earlier cold on-start run.

## Current Model Footprint

The latest validation logs show the following installed model sizes:

| Model | Size |
| --- | ---: |
| `models/musetalkV15/unet.pth` | 3.2G |
| `models/sd-vae/diffusion_pytorch_model.bin` | 320M |
| `models/whisper/pytorch_model.bin` | 145M |
| `models/face-parse-bisent/79999_iter.pth` | 51M |
| `models/dwpose/dw-ll_ucoco_384.pth` | 389M |
| `models/syncnet/latentsync_syncnet.pt` | 1.4G |
| `models/face_detection/s3fd.pth` | 86M |

Evidence in `log_thing`: `2765-2771` and earlier `1464-1470`.

## Bottleneck Ranking

### Current full-stack setup bottlenecks

Ranked by impact in the latest clean setup run:

1. `Install full mmcv` - `6m01s`
2. `Install PyTorch CUDA 12.1 wheel set` - `1m48s`
3. `Download MuseTalk V1.5 weights` - `1m26s`
4. `Install torch-tensorrt pinned stack` - `55s`
5. `Install pinned HLS/api_server dependency set` - `45s`

Conclusion:

- the main dependency bottleneck is still `mmcv`
- the main setup phase bottleneck is still `Build TRT-stagewise experiment venv`
- within that venv build, the dominant sub-phase is `Install avatar-preparation dependencies`

### Cold server-start bottleneck

For a truly cold process start, the bottleneck is different:

1. stagewise TRT VAE warmup
2. specifically the `throughput_record` warmup buckets `8` and `16`

Conclusion:

- setup/install and cold process startup have **different** bottlenecks
- install bottleneck: `mmcv` compile
- runtime startup bottleneck: stagewise TRT warmup before `/health`

## Root Causes

### 1. No published prebuilt `mmcv` wheel for the current env

The current env is:

- Python `3.10`
- torch `2.5.1+cu121`
- CUDA `12.1`

For that environment, the script now explicitly reports that the OpenMMLab wheel index is missing:

- `https://download.openmmlab.com/mmcv/dist/cu121/torch2.5.0/index.html`

That is why `mmcv` falls back to local compilation.

### 2. The venv build discards cache opportunities

Because `--no-cache-dir` is used on most `pip install` calls, the build:

- redownloads packages more often than necessary
- does not retain the built `mmcv` wheel for reuse
- gets less value from a persistent pip cache directory

### 3. The avatar-prep path pulls in more tooling than the server path

This is expected because avatar prep requires the full OpenMMLab stack, but it means the full-stack node pays for:

- `openmim`
- `mmengine`
- `mmcv`
- `mmdet`
- `mmpose`
- associated dependency churn

### 4. Model downloads are still sequential across repos

`HF_MAX_WORKERS=4` helps inside one Hugging Face repo download, but:

- the repos are processed sequentially
- some of the largest files are effectively one file per repo
- `hf_xet` is not currently being used

### 5. Cold-start health is blocked on TRT warmup

The server does not report healthy until FastAPI startup finishes.

Startup creates `ParallelAvatarManager`, which immediately activates and warms the stagewise TRT VAE backend. Because `scripts/vast_server_ctl.sh` waits on `/health`, the whole on-start flow waits for that warmup to finish.

## Highest-Value Improvement Options

### 1. Cache the built `mmcv` wheel and install it locally first

This is the cleanest setup-time win for the current full-stack requirement.

Recommended pattern:

1. build the wheel once for this exact env
2. store it in the repo or another persistent location
3. teach `setup_trt_experiment_env.sh` to prefer the local wheel before source build

Expected impact:

- removes the repeated `~6 minute` `mmcv` source build from future fresh boots on the same environment

### 2. Persist or defer stagewise TRT warmup

This is the biggest cold-start win.

Current problem:

- stagewise TRT warmup happens inside FastAPI startup
- `/health` cannot go green until warmup finishes

Possible fixes:

- persist compiled stagewise engines to disk and reload them
- reduce startup warmup batches for cold boot
- move warmup into a background task after health goes live

Expected impact:

- can remove or greatly reduce the `~10 minute` cold start-to-health penalty on first process start

### 3. Stop throwing away pip caches

Removing `--no-cache-dir` from the normal install path would improve rebuild times, especially when:

- the same node is reused
- a volume is mounted
- a prebuilt local wheelhouse exists

### 4. Replace `openmim` where it is not needed

`mmengine`, `mmdet`, and `mmpose` are available as ordinary wheels, so the dependency strategy could likely be simplified:

- use `pip install` directly for `mmengine`
- use `pip install` directly for `mmdet`
- use `pip install` directly for `mmpose`
- keep `mmcv` as the special case

That would not remove the `mmcv` compile, but it would reduce extra dependency churn in the avatar-prep phase.

### 5. Improve model download path

Potential improvements:

- install `hf_xet`
- pre-stage weights on a volume or image
- optionally parallelize some independent repo downloads

This matters, but it is currently lower priority than `mmcv` and cold TRT warmup.

### 6. Fix `SETUP_CLEAN=1` semantics

This will not make setup faster by itself, but it will make benchmark results more trustworthy because the script will behave the way the name suggests.

## Recommended Order of Work

If the objective is to reduce real first-boot time on a full-stack Vast node, the best order is:

1. cache and reuse `mmcv` locally
2. address cold TRT warmup persistence or health-gating
3. stop discarding pip caches
4. simplify avatar-prep dependency installation away from unnecessary `openmim` churn
5. improve download acceleration and pre-staging
6. clean up `SETUP_CLEAN=1` rebuild semantics

## Bottom Line

There are two separate startup problems:

1. full-stack setup/install time
2. cold process start-to-health time

Right now the bottlenecks are:

- setup/install bottleneck: `mmcv` source build
- cold startup bottleneck: stagewise TRT warmup for batches `8` and `16`

For the latest clean full-stack setup run, the most important number is:

- `Install full mmcv` = `6m01s`

For the earlier true cold full on-start run, the most important number is:

- `Server start-to-health` = `10m21s`

Those are the two biggest levers to improve next.

## Next Implementation Candidate

The next change that is most likely worth doing is:

- add a local `mmcv` wheel path so the script installs a saved wheel before attempting a source build

That is compatible with keeping the node full-stack and should cut the biggest dependency hotspot from repeated boots.
