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

### Phase 1: Vectorized Audio Prompt Building

Goal:

- remove the Python loop in `build_audio_prompts()`

Files:

- `musetalk/utils/audio_processor.py`

Work:

- replace per-frame slicing with a tensorized sliding-window or gather path
- reduce CPU orchestration before generation starts
- keep the implementation easy to benchmark against the current path

Why first:

- it attacks a clear active inefficiency that touches every request
- it does not carry the startup regression risk we just saw with GPU-resident conditioning
- it gives us better information about whether conditioning should ever be revisited on GPU

### Phase 2: GPU-Resident Latent Cycles

Goal:

- stop gathering avatar latents from CPU every generation turn

Files:

- `scripts/api_avatar.py`
- `scripts/hls_gpu_scheduler.py`

Work:

- keep latent cycle tensors on GPU for active avatars
- remove the repeated CPU gather and staging path in scheduler assembly

Why second:

- latent cycles are still a good selective-residency target
- this avoids repeating the exact conditioning experiment that already failed

### Phase 3: Explicit SDPA Attention Path

Goal:

- ensure UNet uses the best attention path this runtime supports

Files:

- `musetalk/models/unet.py`
- `scripts/avatar_manager_parallel.py`

Work:

- verify the current attention processor
- explicitly enable the best SDPA-based path available

Why third:

- worthwhile model-side optimization
- lower risk than another direct output-path or compose-path experiment

### Phase 4: VAE Output Boundary

Goal:

- reduce the cost of the decode-to-CPU handoff

Files:

- `musetalk/models/vae.py`
- `scripts/hls_gpu_scheduler.py`

Work:

- perform more formatting on GPU before transfer
- keep the stable output path easy to revert to

Why fourth:

- still a valid model-path target
- but earlier output-path changes already showed regression risk

### Phase 5: Revisit GPU-Resident Conditioning Only If New Evidence Supports It

Goal:

- reconsider conditioning residency only after earlier phases are measured

Files:

- `musetalk/utils/audio_processor.py`
- `scripts/hls_gpu_scheduler.py`
- `scripts/avatar_manager_parallel.py`

Work:

- only retry if vectorized prompts and the updated timings make a stronger case
- preserve the startup-first scheduler behavior as the non-negotiable baseline
- keep this behind a feature flag if it is ever retried

Caution:

- the direct experiment was already reverted once
- it should not return as a default path without clearly better evidence

### Phase 6: Decision Point

If Phases 1 through 5 still do not move throughput enough, then escalate to the larger architecture path:

- GPU compose
- TensorRT
- ONNX Runtime

## Priority Order

1. vectorized audio prompt building
2. GPU-resident latent cycles
3. explicit SDPA attention optimization
4. VAE output-boundary refinement
5. revisit GPU-resident conditioning only if later evidence justifies it
6. only then consider the larger acceleration branch

## Bottom Line

The next model/GPU optimization push should be:

- **selective**
- **memory-aware**
- **reversible**

We are not trying to move the whole pipeline onto GPU at once.
We are trying to remove the **highest-confidence hot-path waste first**, then selectively keep the right small tensors on GPU only when the measurements support it.
