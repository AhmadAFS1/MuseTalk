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

- best throughput: `51.1 fps` at `batch_size=32`
- max sustainable fps per stream at `8` concurrent: `6.4 fps`

Most important breakdown:

- `PE` cost is negligible
- `UNet` is meaningful but secondary
- `VAE` is the dominant model-side cost

So the backend-acceleration order needs to change. The next serious branch should now start with VAE acceleration, not UNet.

## Current State

The VAE-first backend branch is now beyond repo wiring and has reached a real
alternate-environment TensorRT VAE artifact.

What is done:

- VAE decode backend hook exists in `musetalk/models/vae.py`
- startup/runtime wiring exists in `scripts/avatar_manager_parallel.py`
- backend loader exists in `scripts/trt_runtime.py`
- export script exists in `scripts/tensorrt_export.py`
- benchmark path is backend-aware in `scripts/benchmark_pipeline.py`

What is now also done in the alternate env:

- `/content/py310_trt_exp` has the pinned newer backend family
- VAE TensorRT engine export succeeded there
- exported artifact:
  - `models/tensorrt_altenv_bs32/vae_decoder_trt.ts`
- metadata artifact:
  - `models/tensorrt_altenv_bs32/vae_decoder_trt_meta.json`
- current exported batch support:
  - range `[4, 48]`
  - opt batch `16`
- saved engine size is about `132 MB`
- runtime loading is validated there with fallback disabled
- a broad isolated benchmark now exists there:
  - `benchmark_pipeline_trt_vae_bs32.json`
  - batch set `[4, 8, 16, 32, 48]`
  - best throughput `61.3 fps` at `batch_size=32`
- broad PyTorch comparison already exists:
  - `benchmark_pipeline_results.json`
  - best throughput `51.1 fps` at `batch_size=32`

What is still not done yet:

- no multi-stream `load_test.py` run has been done yet with an active TRT VAE
  backend
- the first single-stream backend-active HLS run is now done
  - `load_test_report.json`
  - eager UNet + TRT VAE
  - `avg_time_to_live_ready_s=3.015`
  - `avg_segment_interval_s=0.196`
  - `max_segment_interval_s=1.512`

Important runtime clarification:

- the validated HLS runtime shape is currently eager UNet + TRT VAE
- a later server run with compiled UNet + TRT VAE reached startup but failed on
  the first HLS generation batch during CUDA graph capture with:
  - `CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED`
  - `CUDA error: operation failed due to a previous error during capture`
- the end-to-end HLS smoke test also showed ffmpeg encoder failure on
  `h264_nvenc`, forcing fallback to CPU `libx264`

Important current split:

- `/content/py310` remains the stable PyTorch server env
- `/content/py310` is still not TRT-ready
- `/content/py310_trt_exp` is TRT-capable for export/runtime/benchmark work
  and now also boots the HLS `api_server.py` path with TRT active
  - WebRTC is still disabled there without `aiortc`
  - avatar preparation in that env is still not the validated path; the
    working server flow uses existing prepared avatars plus the lazy
    preprocessing import in `scripts/api_avatar.py`

Important current interpretation:

- the VAE-only TRT branch is now technically real and materially faster across
  the broad `4..48` batch range
- but it is still far short of the `96 fps` target
- it has now crossed the first `api_server.py` startup milestone in the
  alternate env
- the next real gate is now **correctness**, not more throughput testing

## Critical Correctness Gate

The current active broad-batch TRT VAE artifact in
`models/tensorrt_altenv_bs32` is now known to be visually wrong.

Observed regression:

- the talking-face ROI becomes a flat gray slab in HLS `/wall` output

What the current evidence says:

- `scripts/hls_gpu_scheduler.py` and `scripts/api_avatar.py` still run the
  normal full pipeline
  - UNet predicts latents
  - `vae.decode_latents(...)` produces a face ROI
  - `compose_frame(...)` blends that ROI back into the avatar frame
- direct A/B decode checks show the TRT engine output is already wrong before
  `get_image_blending(...)` ever runs
- the prepared avatar masks and materials under
  `results/v15/avatars/test_avatar` are not the root cause
- the wrapper math in `scripts/tensorrt_export.py` matches PyTorch when run in
  pure PyTorch, so the break is introduced after TensorRT compilation/load

Representative measurements:

- cached/avatar latent decode MAE:
  - about `87.1`
- real UNet-predicted latent decode MAE:
  - about `53.3`
- wrapper control test in pure PyTorch:
  - MAE about `5.3e-05`

Practical meaning:

- the current TRT benchmark win is still useful as a raw speed datapoint
- but the active TRT artifact is **not** yet valid for avatar-quality
  validation
- additional HLS load testing should not be used as the next decision gate
  until this correctness issue is fixed
- the immediate next engineering goal is now TRT correctness gating rather than
  more HLS throughput measurement

The repo now has the first correctness-gate plumbing in place:

- `scripts/validate_vae_backend.py` can directly compare PyTorch vs TRT decode
  outputs on prepared avatar latents
- `scripts/tensorrt_export.py` can validate a saved VAE artifact after export
  and write the result into metadata
- `scripts/trt_runtime.py` can require that validation metadata at activation
  time with `MUSETALK_TRT_REQUIRE_VALIDATION=1`

Current observed status of those guardrails:

- validation against the active broad-batch artifact still fails with
  MAE about `0.3408`
- enabling `MUSETALK_TRT_REQUIRE_VALIDATION=1` now correctly rejects the old
  broad-batch artifact because it has no validation metadata
- an exact-batch FP16 retry now also fails the same correctness gate:
  - artifact dir: `models/tensorrt_fp16_bs4`
  - save format: `torchscript`
  - batch range: `[4, 4]`
  - validation MAE: about `0.340751`

So the current evidence now points more strongly toward:

- the FP16 TRT VAE compile/runtime behavior itself
- not just the old broad dynamic-shape `[4..48]` engine profile
- and not just the `exported_program` serialization route

We have now completed the first two narrowing steps:

1. **in-memory compiled TRT output is already wrong**
2. the first divergent decoder stage is `decoder_mid_block`

New representative findings:

- `scripts/validate_vae_trt_inmemory.py`
  - `batch_size=4`
  - `precision=fp16`
  - TRT in-memory output range: `0.3989..0.5342`, mean `0.4714`
  - MAE vs PyTorch: `0.3407516`
- `scripts/inspect_vae_trt_stages.py`
  - first bad stage: `decoder_mid_block`
  - `scale_post_quant`: exact match
  - `decoder_conv_in`: exact match
  - `decoder_mid_block`: MAE `0.4712`
  - `output_normalize`: exact match when given the same pre-normalized tensor

Execution priority should now be:

1. keep the old monolithic TRT artifact path frozen as broken
2. make the new `trt_stagewise` backend the active correctness branch
3. keep exact-batch validation as the default experiment shape
4. use the repaired `batch_size=4` stagewise path as the first real HLS visual
   checkpoint
5. do not resume larger HLS wall/load comparisons until stagewise TRT passes
   direct decode validation on the relevant buckets

## Current Next Step

The current blocker is no longer export, runtime activation, or the first
isolated benchmark.

The current question is whether the measured gain is large enough to justify
server migration.

Current measured result:

1. `batch_size=16`
   - `50.5 -> 60.2 fps`
   - `253.34 -> 197.75 ms` on VAE full path
2. `batch_size=32`
   - `51.1 -> 61.3 fps`
   - `505.75 -> 396.20 ms` on VAE full path
3. `batch_size=48`
   - `50.9 -> 61.2 fps`
   - `764.93 -> 596.55 ms` on VAE full path
4. implied model-path ceiling at `8` concurrent is still only about
   `7.7 fps/stream`

Current branch status:

- repo-side wiring: done
- alternate-env export: done
- alternate-env runtime load: done
- backend-active matched benchmark: done
- backend-active HLS `api_server.py` startup: done
- backend-active HLS `load_test.py` smoke test: done
- visual correctness of the active TRT artifact: **failed**
- in-memory TRT correctness check: **failed**
- first divergent decoder stage localized: **yes, `decoder_mid_block`**

So the branch should now move to:

- freeze the current broad TRT VAE artifact as **performance-only / untrusted**
- keep the stable PyTorch path as the source of truth for visual validation
- reproduce the TRT-vs-PyTorch decode mismatch in a minimal correctness check
- continue narrowing whether the fault is:
  - TensorRT compilation itself
  - or the current serialization/load route on this stack
- use the new exact-batch result as part of that narrowing:
  - broad dynamic engine: broken
  - exact `batch_size=4` FP16 engine: also broken
- investigate a different backend path only after correctness is restored
  - different TRT export/runtime route
  - and/or a different backend family beyond the current saved TRT engine path

Concrete implementation update:

1. a new backend is now wired into `scripts/trt_runtime.py`:
   - backend name: `trt_stagewise`
   - exact-batch decoder stages compiled independently and cached by batch size
   - `native_group_norm` kept on the PyTorch side during stage compilation
2. first full decode validation with that backend now succeeds at `batch_size=4`
   - script:
     - `scripts/validate_vae_backend.py --backend trt_stagewise`
   - report:
     - `tmp/vae_stagewise_backend_validation_bs4/report.json`
   - MAE:
     - `0.0005082`
   - max abs:
     - `0.0097656`
3. decoder stage probes also improved materially:
   - `decoder_mid_block`: MAE `0.00419`
   - `decoder_up_block_0`: MAE `0.01746`
   - `decoder_postprocess`: MAE `0.000489`
4. current next implementation plan:
   - treat the first `batch_size=4` `/wall` result as visually repaired
   - record the first `concurrency=8` stagewise HLS result at the same bucket:
     - `completed=8`
     - `avg_time_to_live_ready_s=1.631`
     - `avg_segment_interval_s=1.769`
     - `max_segment_interval_s=2.524`
     - interpretation:
       - much better startup and better steady-state pacing than the earlier
         stable PyTorch band
       - still technically throttled by the current `2.0s` max-interval rule
   - important runtime distinction for wider buckets:
     - the active `trt_stagewise` backend does **not** require a regenerated
       reusable TRT artifact just to test `batch_size=8`
     - it compiles and caches exact batch sizes at runtime in
       `scripts/trt_runtime.py`
     - the old serialized TRT export workflow **does** still require explicit
       `--batch-sizes ...` coverage if we ever switch back to that backend
   - validate `trt_stagewise` on `batch_size=8`, `16`, `32`, `48`
     - first concrete gate for `bs8`:
       - `python scripts/validate_vae_backend.py --avatar-id test_avatar --backend trt_stagewise --batch-size 8 --output-dir ./tmp/vae_backend_validation_bs8`
     - after correctness passes, warm the stagewise runtime at `bs8` before
       using a widened HLS scheduler config in live load tests
   - benchmark its throughput against the PyTorch VAE path
   - if it stays correct and faster, widen it into real HLS testing
5. once the widened stagewise buckets pass validation:
   - re-enable runtime enforcement with `MUSETALK_TRT_REQUIRE_VALIDATION=1`
   - then rerun broader HLS wall/load testing against that validated path
6. if mid-block-targeted repair still cannot pass validation:
   - treat this Torch-TensorRT VAE route as blocked for correctness
   - move to a different backend family instead of spending more time on
     scheduler/load testing

## Separate Environment Immediate Next Step

The immediate next execution step is now more of a correctness gate:

1. keep using `/content/py310_trt_exp` only for backend experiments
2. do **not** touch `/content/py310`
3. treat `models/tensorrt_altenv_bs32/vae_decoder_trt.ts` as visually broken
4. keep UNet compile disabled for any remaining TRT repro work
5. keep using the stable PyTorch VAE path for any real lip-sync checks
6. keep the new in-memory and stage-level validation scripts as required gates
   before future load tests
7. if backend work continues after that, move to a different correctness-safe
   backend path rather than assuming this specific TRT artifact is usable

What not to do now:

- do not switch the stable `/content/py310` server to TRT
- do not read the current VAE-only TRT gain as enough to hit the `8 x 12 fps`
  goal by itself
- do not treat the current active TRT artifact as production-valid video output

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
- actual successful engine export: done in the alternate env
- runtime activation in the alternate env: done
- matched isolated benchmark in the alternate env: done
- backend-active HLS `api_server.py` startup: done
- backend-active HLS `load_test.py` smoke test: done
- higher-concurrency backend-active HLS validation: still pending

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

1. host-side HLS pipeline refactor
2. then TensorRT for UNet or another larger backend step
3. TensorRT for VAE follow-on only if a later benchmark justifies it
4. ONNX Runtime fallback
5. VAE output-boundary work only if backend acceleration still leaves a gap
6. GPU compose only if backend acceleration still leaves a gap
7. revisit the rolled-back small experiments only if new profiling justifies it

## Execution-Plan Correction: March 23 CPU/HLS Shift

The latest Threadripper HLS load tests changed the next implementation branch.

The current active `trt_stagewise` VAE path appears visually functional enough
to continue experimentation, but the latest poor `8`-stream run showed:

- `avg_time_to_live_ready_s = 3.469`
- `avg_segment_interval_s = 3.622`
- `max_segment_interval_s = 6.137`
- `avg GPU util = 37.2%`

That means the GPU is underfed. So before another major model/backend branch,
the next implementation phase should now be a host-side HLS pipeline refactor.

### Phase 0: Host-Side HLS Pipeline Refactor

Goal:

- create real headroom by reducing CPU-side backpressure around the shared GPU
  scheduler

Files:

- `scripts/hls_gpu_scheduler.py`
- `scripts/api_avatar.py`
- `musetalk/utils/audio_processor.py`
- `api_server.py`

Work:

- parallelize `_prepare_job()` so avatar load and audio-side prep are not
  front-loaded as one long serial block
- reduce avatar cache-miss cost by parallel or lazy frame/mask loading
- replace per-chunk `ffmpeg` spawn with a persistent encoder / segmenter path
- refactor compose so more CPU cores can work without just increasing thread
  contention
- converge the older direct live-streaming routes onto the shared scheduler
  model where practical

Why first:

- the latest regression is no longer explained by model throughput alone
- current live HLS behavior is dominated by prep / compose / encode /
  queueing behavior
- another backend export will not fix a GPU that is already waiting for work

Current implementation status:

- first refactor slice now landed:
  - `musetalk/utils/audio_processor.py`
    - batched feature-extractor path
    - batched Whisper-segment encode path
    - vectorized prompt construction
  - `scripts/hls_gpu_scheduler.py`
    - concurrent avatar load + audio feature extraction in `_prepare_job()`
    - overlapped idle-frame preload
  - `scripts/api_avatar.py`
    - parallel cache-miss frame/mask loading
- first measured `concurrency=8` result after that slice:
  - `avg_time_to_live_ready_s=1.760`
  - `avg_segment_interval_s=1.733`
  - `max_segment_interval_s=2.535`
  - `avg GPU util ~= 82.06%`
  - practical meaning:
    - this refactor direction is validated
    - the severe GPU-underfed regression is largely recovered
    - the remaining work is now about shaving the tail, not rescuing a broken path
- later March 24 ramp results now clarify the next milestone:
  - `concurrency=6`
    - `avg_time_to_live_ready_s=1.342`
    - `avg_segment_interval_s=1.294`
    - `max_segment_interval_s=2.032`
    - practical meaning:
      - this is the first practical realtime milestone on the current branch
      - the warning threshold is only missed by about `32ms`
  - `concurrency=7`
    - `avg_segment_interval_s=1.516`
    - `max_segment_interval_s=2.530`
  - repeated `concurrency=8`
    - `avg_segment_interval_s=1.733-1.736`
    - `max_segment_interval_s=2.524-2.535`
    - practical meaning:
      - `7` and `8` are now in the same batch-4 saturation band
      - the next gains should come from lowering tail latency, not expecting a
        different scheduler regime automatically at `7`
- later 64-core worker-scale check (`prep/compose/encode = 12/12/12`)
  - `concurrency=8`
    - `avg_segment_interval_s=1.827`
    - `max_segment_interval_s=2.533`
    - `avg GPU util ~= 76.83%`
  - `concurrency=10`
    - `avg_segment_interval_s=2.251`
    - `max_segment_interval_s=3.531`
    - `avg GPU util ~= 80.33%`
  - practical meaning:
    - the larger worker profile did **not** create a new tier on the 64-core box
    - keep the `8/8/8` worker baseline and move the next effort into encode /
      compose structure instead
- later first widened live `bs8` scheduler experiment on March 25:
  - `concurrency=1`, `batch_size=8`
    - `avg_time_to_live_ready_s=1.006`
    - `avg_segment_interval_s=0.192`
    - `avg GPU memory used ~= 13742 MB`
  - `concurrency=8`, `batch_size=8`
    - `avg_time_to_live_ready_s=1.947`
    - `avg_segment_interval_s=1.513`
    - `max_segment_interval_s=2.531`
    - `wall_time_s=28.4`
    - `avg GPU util ~= 82.87%`
    - `avg GPU memory used ~= 13821 MB`
  - practical meaning:
    - this is the first sign that widening the live stagewise bucket can
      materially improve steady-state pacing
    - the tail still stayed around `2.53s`, so this is not a full solution yet
    - the first test forced `fixed_batch_sizes=[8]`, which is too blunt for the
      long-term config because even small turns pad to `8`
    - the next live config to measure should be:
      - `HLS_SCHEDULER_MAX_BATCH=8`
      - `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8`
      - `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4`
- later widened `max_batch=16` branch on March 25 set the current best
  `concurrency=8` average-throughput result:
  - server-side shape:
    - `HLS_SCHEDULER_MAX_BATCH=16`
    - `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16`
    - `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4`
    - workers still `8/8/8`
  - request `batch_size=8`
    - `avg_time_to_live_ready_s=2.197`
    - `avg_segment_interval_s=1.408`
    - `max_segment_interval_s=2.525`
    - `wall_time_s=26.4`
    - `avg GPU util ~= 85.21%`
    - `avg GPU memory used ~= 23922 MB`
  - same server branch with request `batch_size=4`
    - `avg_segment_interval_s=1.423`
    - `max_segment_interval_s=2.046`
    - `wall_time_s=26.4`
  - practical meaning:
    - the widened total batch budget is real and beneficial
    - request `batch_size=8` is the new best average-throughput point
    - request `batch_size=4` keeps the better tail on the same branch
    - the next implementation work should now assume the branch is GPU-dense
      enough and focus on shrinking tail latency without losing the new average
      throughput gain
- still pending inside Phase 0:
  - deeper encode pipeline work beyond audio-sidecar reuse
  - deeper compose refactor
  - convergence of older direct live-serving paths

Current implementation update:

- the next host-side slice now landed in code:
  - `scripts/api_avatar.py`
    - reusable AAC sidecar prep
    - chunk muxing attempts `-c:a copy` first, then falls back to inline AAC
    - compose now reuses a cached per-cycle blending plan
  - `musetalk/utils/blending.py`
    - cached blend-plan helpers now hold static crop geometry and alpha
  - `scripts/hls_gpu_scheduler.py`
    - sidecar prep overlaps with HLS request prep
    - sidecar cleanup follows request cleanup
- this slice has passed targeted smoke validation, but `load_test.py`
  confirmation is still pending after it

## Bottom Line

The next implementation push should be:

- **host-pipeline-focused first**
- **benchmark-driven**
- **reversible**

We already proved that the repaired TRT path can materially improve the model
side, but the latest Threadripper HLS runs now show the GPU waiting on the host
pipeline. So the next serious branch should not be "more thread caps" and it
should not immediately be "another export first." It should be a host-side HLS
pipeline refactor, followed by another backend branch only after that refactor
is measured.

That measurement is now partially in: the first host-pipeline slice recovered
the path from the `3.469 / 3.622 / 6.137` regression band back to about
`1.760 / 1.733 / 2.535`, and the later March 24 ramp results now show a
practical `concurrency=6` realtime milestone with `1.294 / 2.032` cadence.
So the right next move is still to continue Phase 0, not to pivot away from it
yet.
