# Current Start Param Reference

This file explains what the current launch params in [`start_params.md`](./start_params.md) actually do in the codebase today.

Scope:

- stable PyTorch HLS start path
- March 2026 code state
- practical meaning, not just the variable name

Important framing:

- some knobs raise the actual model-throughput ceiling
- some only change queueing or startup behavior
- some are mostly operational/debug toggles
- one documented knob, `HLS_PERSISTENT_SEGMENTER`, appears to be a legacy holdover and is not currently read by the live code path

## Param Groups

### PyTorch Runtime

#### `unset PYTORCH_CUDA_ALLOC_CONF`

What it does:

- clears any previously exported PyTorch CUDA allocator override from the shell before starting the server

What it does **not** do:

- it is not read by MuseTalk code directly
- it does not change scheduling logic on its own

Practical meaning:

- this keeps the server on PyTorch's default allocator behavior unless another script sets a new allocator config later
- it is a shell hygiene step, not a MuseTalk tuning lever

Recommendation:

- keep it unset for the stable baseline unless you are intentionally testing allocator behavior

### Model Compile And Warmup

These settings are read in [`scripts/avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py).

#### `MUSETALK_COMPILE=1`

Code path:

- checked in [`avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L318)

What it does:

- enables the `compile_models()` path
- allows `torch.compile(...)` to be attempted for UNet and VAE

Real effect:

- this is one of the few knobs here that can raise the raw model-path ceiling
- it can also fail or behave differently across environments

Recommendation:

- keep enabled on the current stable PyTorch path

#### `MUSETALK_COMPILE_UNET=1`

Code path:

- checked in [`avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L338)

What it does:

- allows the UNet to be compiled with `torch.compile`

Real effect:

- only affects the UNet path
- if disabled, the UNet stays eager even when `MUSETALK_COMPILE=1`

Recommendation:

- keep enabled for the stable PyTorch path

#### `MUSETALK_COMPILE_VAE=1`

Code path:

- checked in [`avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L361)

What it does:

- allows the VAE to be compiled with `torch.compile`
- skipped automatically when an alternate decode backend like TRT is active

Real effect:

- affects the PyTorch VAE path only
- on the current stable non-TRT path, this is still part of the baseline

Recommendation:

- keep enabled on the stable PyTorch path
- set to `0` only when explicitly testing a different VAE backend

#### `MUSETALK_COMPILE_MODE=reduce-overhead`

Code path:

- resolved in [`avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L206)

What it does:

- tells `torch.compile` which mode to try first

Real effect:

- changes the compile/runtime tradeoff
- `reduce-overhead` is the current conservative choice for a live server

Recommendation:

- keep `reduce-overhead` for the stable start path

#### `MUSETALK_COMPILE_TRACEBACK=1`

Code path:

- used in compile failure logging in [`avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L243)

What it does:

- prints full tracebacks when compile or warmup fails

Real effect:

- debug visibility only
- not a throughput knob

Recommendation:

- fine to leave on during active investigation
- can be turned off later if logs become too noisy

#### `MUSETALK_COMPILE_WARMUP_BATCHES=4,8,16,32,48`

Code path:

- parsed in [`avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L230)
- used to warm compiled UNet/VAE in [`avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L252) and [`avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L279)

What it does:

- pre-warms the compiled models on the listed batch shapes

Real effect:

- reduces first-hit compile surprises during live traffic
- should stay aligned with the scheduler's fixed batch buckets

Recommendation:

- keep this matched with `HLS_SCHEDULER_FIXED_BATCH_SIZES`

#### `MUSETALK_WARM_RUNTIME=1`

Code path:

- checked in [`avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L392)

What it does:

- warms Whisper + positional encoding runtime paths before the first live request

Real effect:

- helps cold-start behavior more than steady-state throughput

Recommendation:

- keep enabled for live serving

### Avatar Cache

These settings are read in [`scripts/avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L50).

#### `AVATAR_CACHE_MAX_AVATARS=0`

What it does:

- sets the avatar-count limit for the in-memory avatar cache

Real effect:

- in this codebase, `0` is used as the "do not limit by avatar count" mode
- memory is still constrained by the memory cap

Recommendation:

- keep `0` when you want the memory budget, not avatar count, to be the controlling limit

#### `AVATAR_CACHE_MAX_MEMORY_MB=12000`

What it does:

- caps how much RAM the avatar cache should consume

Real effect:

- larger values reduce cache eviction pressure
- too small a value causes more avatar reload churn

Recommendation:

- `12000` is a practical large-cache value for this current workload

#### `AVATAR_CACHE_TTL_SECONDS=3600`

What it does:

- sets how long cached avatars can remain idle before TTL cleanup considers them stale

Real effect:

- mostly operational behavior, not a direct throughput lever

Recommendation:

- `3600` is reasonable for long-running service use

### HLS Scheduler Batch Shaping

These values are passed from [`api_server.py`](/content/MuseTalk/api_server.py#L419) into [`scripts/hls_gpu_scheduler.py`](/content/MuseTalk/scripts/hls_gpu_scheduler.py).

#### `HLS_SCHEDULER_MAX_BATCH=48`

Code path:

- passed in [`api_server.py`](/content/MuseTalk/api_server.py#L423)
- stored in [`hls_gpu_scheduler.py`](/content/MuseTalk/scripts/hls_gpu_scheduler.py#L109)

What it does:

- caps the total combined frame batch the shared HLS scheduler may send through one GPU turn

Real effect:

- real throughput lever
- too low wastes GPU opportunity
- too high can increase latency, jitter, and VRAM pressure

Current finding:

- `48` is the best practical ceiling found so far on this RTX 3090 path

#### `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,32,48`

Code path:

- read in [`hls_gpu_scheduler.py`](/content/MuseTalk/scripts/hls_gpu_scheduler.py#L1471)
- also used for compile warmup defaults in [`avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L103)

What it does:

- defines the preferred padded batch buckets the scheduler and compile warmup should align to

Real effect:

- helps keep GPU work on known, warmed shapes instead of arbitrary one-off shapes

Recommendation:

- keep it aligned with `MUSETALK_COMPILE_WARMUP_BATCHES`

#### `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4`

Code path:

- passed in [`api_server.py`](/content/MuseTalk/api_server.py#L424)
- stored in [`hls_gpu_scheduler.py`](/content/MuseTalk/scripts/hls_gpu_scheduler.py#L110)

What it does:

- controls the first-round allocation size for startup jobs in the shared scheduler

Real effect:

- startup fairness knob
- limited by each stream's own requested `batch_size`

Practical caveat:

- this cannot force a `batch_size=4` stream to behave like a bigger stream

#### `HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS=999`

Code path:

- passed in [`api_server.py`](/content/MuseTalk/api_server.py#L425)
- stored in [`hls_gpu_scheduler.py`](/content/MuseTalk/scripts/hls_gpu_scheduler.py#L111)

What it does today:

- mostly a historical/diagnostic value that still appears in config and stats

Real effect:

- the current scheduler does not appear to use it as a meaningful gate on fill behavior
- this is effectively near-no-op territory in the current code

Recommendation:

- leave it alone unless the scheduler logic is intentionally changed again

### Startup Chunking

These values are passed from [`api_server.py`](/content/MuseTalk/api_server.py#L430) into the HLS scheduler.

#### `HLS_STARTUP_CHUNK_DURATION_SECONDS=0.5`

What it does:

- sets the duration of the early startup chunk size target

Real effect:

- smaller startup chunks help first playback readiness
- too small can create more encode churn

Recommendation:

- `0.5` is the current low-latency startup setting

#### `HLS_STARTUP_CHUNK_COUNT=1`

What it does:

- controls how many early chunks use the startup-sized target before normal chunk sizing takes over

Real effect:

- startup-latency knob, not a raw model-throughput knob

Recommendation:

- `1` is the current minimal startup bias

### CPU Worker Pools

These values are passed from [`api_server.py`](/content/MuseTalk/api_server.py#L426) into [`hls_gpu_scheduler.py`](/content/MuseTalk/scripts/hls_gpu_scheduler.py).

#### `HLS_PREP_WORKERS=8` or `12`

Code path:

- exact pool size used in [`hls_gpu_scheduler.py`](/content/MuseTalk/scripts/hls_gpu_scheduler.py#L114)

What it does:

- sets the thread pool size for HLS job preparation work

Real effect:

- can reduce prep queueing
- does not change the model compute ceiling

Current finding:

- the Threadripper host improved with a modest increase to `12`
- pushing prep too high together with other worker increases did not keep improving the system

#### `HLS_COMPOSE_WORKERS=8` or `10`

Code path:

- requested in [`api_server.py`](/content/MuseTalk/api_server.py#L427)
- effective size computed in [`hls_gpu_scheduler.py`](/content/MuseTalk/scripts/hls_gpu_scheduler.py#L123)

What it does:

- sizes the CPU compose thread pool used for frame blending / frame assembly

Important nuance:

- the actual pool size is:
  - `max(6, requested, min(10, cpu_count // 2))`
- that means high-core machines already get a floor near `10`

Real effect:

- this can reduce compose queue wait
- it can also oversubscribe CPU and memory traffic if pushed too high

Current finding:

- on the Threadripper host, `10` was better than the default `8`
- a more aggressive `12` regressed startup/tail behavior

#### `HLS_ENCODE_WORKERS=8` or `10`

Code path:

- requested in [`api_server.py`](/content/MuseTalk/api_server.py#L428)
- effective size computed in [`hls_gpu_scheduler.py`](/content/MuseTalk/scripts/hls_gpu_scheduler.py#L136)

What it does:

- sizes the worker pool that submits chunk encode work

Important nuance:

- the actual pool size follows the same floor logic as compose:
  - `max(6, requested, min(10, cpu_count // 2))`

Real effect:

- can reduce encode queue wait
- too many workers can increase ffmpeg/NVENC contention

Current finding:

- `10` was a better large-core tuning point than the default `8`
- `12` did not continue improving the live path

#### `HLS_MAX_PENDING_JOBS=24`

Code path:

- passed in [`api_server.py`](/content/MuseTalk/api_server.py#L429)
- enforced in [`hls_gpu_scheduler.py`](/content/MuseTalk/scripts/hls_gpu_scheduler.py#L195)

What it does:

- limits how many HLS jobs may be queued/preparing/active before new stream submissions are rejected

Real effect:

- admission control and backlog behavior
- not a speed knob by itself

Recommendation:

- treat this as queue protection, not a throughput multiplier

### Chunk Encode Backend

These values are read in [`scripts/api_avatar.py`](/content/MuseTalk/scripts/api_avatar.py).

#### `HLS_CHUNK_VIDEO_ENCODER=h264_nvenc`

Code path:

- read in [`api_avatar.py`](/content/MuseTalk/scripts/api_avatar.py#L1053)

What it does:

- chooses the preferred chunk video encoder for ffmpeg

Real effect:

- `h264_nvenc` uses the GPU video encoder path for chunk creation
- the code still falls back to `libx264` if NVENC fails

Recommendation:

- keep `h264_nvenc` for the current live HLS path

#### `HLS_PERSISTENT_SEGMENTER=0`

What the docs used to mean:

- this once referred to an experimental persistent NVENC segmenter path

What I can verify in the current code:

- I do not see the current live code path reading this env var
- it appears in docs and experiment history, but not in the active server/scheduler/avatar encode code

Practical meaning today:

- treat it as legacy documentation state, not a live tuning knob

Recommendation:

- leaving it at `0` is harmless
- do not expect changing it to affect the current code unless that persistent segmenter path is reintroduced

## Best-Known Current Start Blocks

### General Stable PyTorch Path

Use the baseline block from [`start_params.md`](./start_params.md).

### Large-Core Threadripper Variant

Current best-known tuning from the March 22, 2026 comparison:

- `HLS_PREP_WORKERS=12`
- `HLS_COMPOSE_WORKERS=10`
- `HLS_ENCODE_WORKERS=10`

Why:

- this improved the Threadripper host's live path compared with the default `8/8/8`
- a more aggressive `16/12/12` variant regressed startup and worst-case segment timing again

## Related But Optional Knobs

These are not in the current default start block, but they exist nearby in the code.

### `HLS_CHUNK_ENCODER_PRESET`

Code path:

- read in [`api_avatar.py`](/content/MuseTalk/scripts/api_avatar.py#L58)

Meaning:

- ffmpeg encoder preset for chunk encoding

### `HLS_CHUNK_ENCODER_TUNE`

Code path:

- read in [`api_avatar.py`](/content/MuseTalk/scripts/api_avatar.py#L59)

Meaning:

- ffmpeg encoder tune for the NVENC path

### `HLS_CHUNK_ENCODER_QP`

Code path:

- read in [`api_avatar.py`](/content/MuseTalk/scripts/api_avatar.py#L61)

Meaning:

- constant-QP value for the NVENC chunk path

### `HLS_CHUNK_ENCODER_CRF`

Code path:

- read in [`api_avatar.py`](/content/MuseTalk/scripts/api_avatar.py#L69)

Meaning:

- CRF value for the `libx264` fallback path

### `LIVE_MAX_CONCURRENT_GENERATIONS`

Code path:

- read in [`avatar_manager_parallel.py`](/content/MuseTalk/scripts/avatar_manager_parallel.py#L41)

Meaning:

- affects the `GPUMemoryManager` logical live-generation slot budget

Important caveat:

- this is not the main knob for the shared HLS GPU scheduler path being load-tested here

## Bottom Line

For the current stable HLS start block:

- the real throughput levers are mainly:
  - `MUSETALK_COMPILE`
  - `MUSETALK_COMPILE_UNET`
  - `MUSETALK_COMPILE_VAE`
  - `HLS_SCHEDULER_MAX_BATCH`
- the main startup/fairness levers are:
  - `HLS_SCHEDULER_STARTUP_SLICE_SIZE`
  - `HLS_STARTUP_CHUNK_DURATION_SECONDS`
  - `HLS_STARTUP_CHUNK_COUNT`
- the main CPU queueing levers are:
  - `HLS_PREP_WORKERS`
  - `HLS_COMPOSE_WORKERS`
  - `HLS_ENCODE_WORKERS`
- the main operational levers are:
  - avatar cache settings
  - `HLS_MAX_PENDING_JOBS`
  - ffmpeg encoder selection

That is why a machine can have the same RTX 3090 and still behave differently: the live path is not GPU-only, and some of the "performance" knobs are really CPU queueing knobs.
