# MuseTalk Selective GPU Execution Plan

## Core Principle

We should **not** put everything on GPU.

That would risk reducing headroom for the actual generation path, and we already know this system is sensitive to:

- VRAM pressure
- NVENC resource pressure
- downstream pipeline stalls

What *does* make sense is **selective GPU residency** for small tensors that are:

- reused frequently
- copied every generation turn today
- cheap in memory compared with the models and decoded frames

## What Should Go On GPU

These are the best candidates:

- per-avatar latent cycles
- per-request conditioning chunks, but only as a later reconsideration if new evidence justifies it

Why:

- they are reused heavily
- they are relatively small
- moving them to GPU can remove repeated CPU staging and host-to-device copies

Approximate footprint:

- conditioning: roughly `~4.4 MB` to `~11 MB` per request depending on frame count
- latent cycle: roughly `~18.75 MB` for a `1200`-frame avatar cycle

That is very different from keeping full decoded RGB frames or full encode surfaces on GPU.

## Recent Update

We already tested direct GPU-resident conditioning in the shared HLS scheduler and reverted it.

Observed result at `concurrency=8`, `batch_size=4`, `playback_fps=24`, `musetalk_fps=12`:

- average segment interval stayed about the same at `1.961-1.965s`
- max segment interval stayed about the same at `3.08-3.21s`
- startup fairness regressed:
  - one stream came live around `3.0s`
  - the rest clustered around `5.0s`
  - `avg_time_to_live_ready_s` regressed to about `4.78s`

So we are not treating GPU-resident conditioning as Phase 1 anymore. It remains a possible later revisit, not the current lead bet.

We also tested vectorized audio-prompt building and rolled it back as a throughput change.

Observed result at `concurrency=8`, `batch_size=4`, `playback_fps=24`, `musetalk_fps=12`:

- `avg_segment_interval_s = 2.0`
- `max_segment_interval_s = 3.225`
- `avg_time_to_live_ready_s = 4.772`

So that refactor was valid as a cleanup, but it did not materially improve throughput or startup. We are not treating it as the active lead bet anymore either.

We also tested GPU-resident latent cycles and rolled that back as a throughput change.

Observed result at `concurrency=8`, `batch_size=4`, `playback_fps=24`, `musetalk_fps=12`:

- run 1:
  - `avg_segment_interval_s = 1.971`
  - `max_segment_interval_s = 3.143`
  - `avg_time_to_live_ready_s = 4.774`
- run 2:
  - `avg_segment_interval_s = 1.99`
  - `max_segment_interval_s = 3.119`
  - `avg_time_to_live_ready_s = 4.774`

So latent-cycle residency was another reasonable selective-GPU experiment, but it also failed to materially improve throughput or startup. That means the next branch should stop assuming selective residency is the main answer.

We also tested explicit SDPA attention-path tuning and rolled that back as a throughput change.

Observed result across repeated `concurrency=8`, `batch_size=4`, `playback_fps=24`, `musetalk_fps=12` runs:

- `avg_segment_interval_s` stayed in the same familiar `~1.97-2.04s` band
- `max_segment_interval_s` stayed in the same familiar `~3.10-3.22s` band
- `avg_time_to_live_ready_s` stayed in the same familiar `~4.15-4.77s` band

So the SDPA branch was another reasonable model-path experiment, but it also failed to materially improve throughput. That means the selective/incremental PyTorch-path branch is now sufficiently exhausted, and the next serious move should be the larger backend-acceleration path in `current_model_backend_acceleration_plan.md`.

We now also have a clean isolated model-path benchmark from `scripts/benchmark_pipeline.py`.

Observed result:

- best throughput: `51.0 fps` at `batch_size=16`
- max sustainable fps per stream at `8` concurrent: `6.4 fps`

Most important breakdown:

- `PE` cost is negligible
- `UNet` is meaningful but secondary
- `VAE` is the dominant model-side cost

So the backend-acceleration order needs to change. The next serious branch should now start with VAE acceleration, not UNet.

## Current State

The VAE-first backend branch is now partly implemented in code, but not yet activated in practice.

What is done:

- VAE decode backend hook exists in `musetalk/models/vae.py`
- startup/runtime wiring exists in `scripts/avatar_manager_parallel.py`
- backend loader exists in `scripts/trt_runtime.py`
- export script exists in `scripts/tensorrt_export.py`
- benchmark path is backend-aware in `scripts/benchmark_pipeline.py`

What is now also done locally:

- `torch-tensorrt==1.4.0` is installed
- `tensorrt_bindings==8.6.1` is installed
- `tensorrt_libs==8.6.1` is installed
- repo-root compatibility shim `tensorrt.py` exists so local repo scripts can import `tensorrt` in this environment

What is still not done yet:

- no benchmark has been run with an active TensorRT VAE backend
- no `load_test.py` run has been done with an active TensorRT VAE backend

What is now done:

- VAE TensorRT engine export succeeded
- exported artifact:
  - `models/tensorrt/vae_decoder_trt.ts`
- metadata artifact:
  - `models/tensorrt/vae_decoder_trt_meta.json`
- reported compile time was about `807.4s`
- saved engine size is about `141 MB`

That means this branch did reach engine export, but it still has **not** reached a trustworthy runtime-validated TensorRT VAE path on the current environment.

## Current Next Step

The current blocker is no longer installation.

The current missing validation is runtime proof:

- a benchmark attempt was run with TensorRT requested
- but backend activation failed and the benchmark fell back to PyTorch VAE
- throughput therefore stayed unchanged at about `50.9 fps`
- after the runtime loader was patched, the exporter was tightened to require a full TRT compile
- that full compile then failed on unsupported operators in the current stack:
  - `aten::scaled_dot_product_attention`
  - `aten::group_norm`

So the branch status is now:

- repo-side wiring: done
- backend packages: installed
- engine artifact: exported once, but not trustworthy for validation
- runtime acceleration: not reliably activatable on the current environment
- throughput result: unchanged because the attempted TRT benchmark actually ran on PyTorch fallback

That means we should now treat the **current environment** TensorRT VAE branch as blocked and move to the next execution branch:

- create a separate TensorRT-focused environment
- retry export and runtime validation there
- only then reconsider a backend-active benchmark and `load_test.py`

## Separate Environment Immediate Next Step

The first setup attempt for `/content/py310_trt_exp` is no longer theoretical.
We now know the immediate blockers and the next exact move.

What happened:

- the venv was created successfully
- the first validation import failed because `torch` had not been installed yet
- the first real install attempt then failed on disk pressure:
  - `OSError: [Errno 28] No space left on device`
- a later unpinned backend install attempt also tried to pull the latest
  `torch-tensorrt 2.10.0` family and `tensorrt-cu13`

So the immediate next execution step is now:

1. delete the half-installed `/content/py310_trt_exp`
2. clear `/root/.cache/pip`
3. recreate the venv
4. install the new backend stack with `--no-cache-dir`
5. pin the backend family explicitly

Current recommended install family for the retry:

- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchaudio==2.5.1`
- `cu121`
- `torch-tensorrt==2.5.0`

What not to do on the retry:

- do not run unpinned `pip install torch-tensorrt tensorrt`
- do not proceed to MuseTalk runtime dependency installation before the backend
  import validation succeeds

## What Should Not Go On GPU Yet

These are not the next target:

- full RGB frame cycles
- full mask/frame caches
- persistent per-stream GPU encode state
- “everything on GPU”

Those would create much larger residency costs and are more likely to reduce generation headroom than help throughput.

## Important Clarification

`h264_nvenc` using the GPU does **not** mean every extra GPU-resident tensor is automatically bad.

The real distinction is:

- **good GPU residency**: small persistent tensors that remove repeated copies
- **bad GPU residency**: large persistent buffers that crowd out generation or create resource contention

This plan only targets the first category.

## Concentrated Execution Plan

### Phase 0: Guardrails First

Goal:

- make every optimization measurable and reversible

Files:

- `scripts/hls_gpu_scheduler.py`
- `scripts/avatar_manager_parallel.py`

Work:

- add feature flags for each optimization
- log peak VRAM and `avg_gpu_batch` before and after each phase
- keep easy rollback switches

Success gate:

- no OOM
- no visual regressions
- peak VRAM remains comfortably below the cliff

### Phase 1: Hand Off To The Backend Acceleration Plan

Goal:

- move from the exhausted small-optimization branch to the larger backend-acceleration branch

Files:

- `current_model_backend_acceleration_plan.md`
- `scripts/benchmark_pipeline.py`

Work:

- run the baseline benchmark from `current_model_backend_acceleration_plan.md`
- confirm the true UNet / VAE / transfer breakdown before backend integration
- use that as the entrypoint into TensorRT or ONNX Runtime work

Why first:

- the smaller model/GPU-path experiments have now all failed to create a meaningful throughput shift
- this is the cleanest way to stop guessing and move to the next class of optimization
- it matches the current recommendation in the broader planning doc

### Phase 2: TensorRT For VAE

Goal:

- accelerate the dominant remaining model component using a new backend

Files:

- `musetalk/models/vae.py`
- `scripts/avatar_manager_parallel.py`
- `scripts/tensorrt_export.py`
- `scripts/trt_runtime.py`

Work:

- export and benchmark a TensorRT VAE decoder path
- integrate it with runtime fallback
- measure whether it finally shifts the throughput ceiling

Current status:

- runtime fallback integration: done
- export script scaffolding: done
- backend packages: installed
- actual successful engine export: happened once in the current environment
- runtime activation on the current environment: failed
- strict full compile on the current environment: blocked by unsupported operators
- backend-active benchmark / load-test validation: still pending in a **separate** TensorRT-focused environment

Why second:

- the benchmark showed VAE is the dominant model-side bottleneck
- this is now the highest-upside remaining path after the smaller PyTorch-path experiments failed

### Phase 3: TensorRT For UNet

Goal:

- accelerate the second major model component if VAE TensorRT alone is not enough

Files:

- `musetalk/models/unet.py`
- `scripts/avatar_manager_parallel.py`
- `scripts/tensorrt_export.py`
- `scripts/trt_runtime.py`

Work:

- export and benchmark a TensorRT UNet path
- integrate it with runtime fallback
- measure whether combined backend acceleration creates real headroom

Why third:

- still cleaner than another fragile output/composite rewrite
- remains a higher-upside branch than revisiting rolled-back selective-residency ideas
- follows the benchmark-driven VAE-first order

### Phase 4: ONNX Runtime Fallback

Goal:

- keep a second backend-acceleration path ready if TensorRT export/runtime integration gets blocked

Files:

- `scripts/onnx_export.py`
- `scripts/ort_runtime.py`
- `scripts/avatar_manager_parallel.py`
- `scripts/hls_gpu_scheduler.py`

Work:

- export ONNX models
- wire a runtime path with fallback
- benchmark against PyTorch and TensorRT

Why fourth:

- it is the practical fallback behind the preferred TensorRT path
- it still has more remaining upside than returning to the rolled-back small-model-path experiments

### Phase 5: VAE Output Boundary Only If Backend Acceleration Still Leaves A Gap

Goal:

- reduce the cost of the decode-to-CPU handoff only if backend acceleration still leaves a meaningful gap

Files:

- `musetalk/models/vae.py`
- `scripts/hls_gpu_scheduler.py`
- `scripts/api_avatar.py`

Work:

- perform more formatting on GPU before transfer
- keep the stable output path easy to revert to
- treat this as follow-on work, not the lead branch

Caution:

- earlier output-path changes already showed visual-regression risk
- this should remain behind the backend-acceleration branch, not replace it

### Phase 6: GPU Compose Only If Backend Acceleration Still Leaves A Gap

Goal:

- move the compose path closer to the model path only if backend acceleration still leaves a meaningful gap

Files:

- `scripts/api_avatar.py`
- `musetalk/utils/blending.py`

Work:

- prototype a GPU-native resize/blend path
- keep a safe CPU fallback because this path has visual-regression risk
- treat this as a later architecture-side lever, not the lead branch

Caution:

- this path touches visual correctness directly
- it should remain behind the cleaner backend-acceleration path

### Phase 7: Revisit Rolled-Back Small Experiments Only If New Profiling Creates A Stronger Case

If Phases 1 through 6 still do not move throughput enough, only then reconsider the rolled-back small experiments with fresh evidence:

- vectorized audio prompts
- GPU-resident conditioning
- GPU-resident latent cycles

## Priority Order

1. baseline benchmark from `current_model_backend_acceleration_plan.md`
2. TensorRT for VAE
3. TensorRT for UNet
4. ONNX Runtime fallback
5. VAE output-boundary work only if backend acceleration still leaves a gap
6. GPU compose only if backend acceleration still leaves a gap
7. revisit the rolled-back small experiments only if new profiling justifies it

## Bottom Line

The next model/GPU optimization push should be:

- **backend-focused**
- **benchmark-driven**
- **reversible**

We already tried the highest-confidence small PyTorch-path adjustments and they did not move the mission enough. The benchmark then showed the current PyTorch model path tops out around `51 fps`, with VAE as the dominant cost. The next push should therefore be a larger backend acceleration branch, starting with VAE TensorRT.
