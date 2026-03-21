# MuseTalk Model And GPU Optimization Findings

## Scope

This document captures the current findings from a repo-wide throughput review focused on:

- the active HLS inference path
- the `scripts/` runtime code
- the inference-side model code under `musetalk/`
- the root entrypoints that actually drive live streaming

It is separate from `model_optimization_plan.md`, which is a broader implementation-planning document.

## Bottom Line

The current throughput ceiling is no longer mainly a scheduler-knob problem.

The remaining hot path is:

- `api_server.py`
- `scripts/avatar_manager_parallel.py`
- `scripts/hls_gpu_scheduler.py`
- `scripts/api_avatar.py`
- `musetalk/utils/audio_processor.py`
- `musetalk/models/vae.py`
- `musetalk/models/unet.py`

The biggest remaining inefficiencies are:

1. Whisper features are encoded on GPU and then moved back to CPU.
2. Audio prompts are rebuilt in Python frame-by-frame.
3. The HLS scheduler still stages conditioning and latents on CPU before copying them to GPU every turn.
4. VAE output immediately round-trips to CPU NumPy.
5. Frame compose is still CPU/OpenCV.
6. Segment creation still launches a fresh `ffmpeg` process per chunk.

This means the best remaining model/GPU opportunities are mostly about:

- keeping more of the hot data resident on GPU
- removing repeated host-to-device copies
- vectorizing CPU orchestration that still sits around the model path

## Active Runtime Path

### Root Entry Points

- `api_server.py`
  - Active server entrypoint for HLS and WebRTC.
  - Creates `ParallelAvatarManager`, `HlsSessionManager`, and `HLSGPUStreamScheduler`.
- `load_test.py`
  - Test harness and throttling detector.
  - Important for interpreting results, but not a throughput bottleneck itself.
- `app.py`
  - Demo path, not the primary HLS path under current load testing.

### Scripts That Matter For HLS Throughput

- `scripts/avatar_manager_parallel.py`
  - Model loading, dtype setup, `torch.compile`, Whisper init, avatar cache access.
- `scripts/hls_gpu_scheduler.py`
  - Shared batched GPU generation loop for concurrent HLS streams.
- `scripts/api_avatar.py`
  - Avatar materials, compose path, direct streaming path, per-chunk encoding.
- `scripts/concurrent_gpu_manager.py`
  - GPU lease/accounting helper. Operationally relevant, but not the main throughput ceiling.
- `scripts/hls_session_manager.py`
  - HLS manifest/session lifecycle. Important for delivery, but not the main model/GPU bottleneck.

### Scripts That Are Not The Next Optimization Target

- `scripts/session_manager.py`
- `scripts/webrtc_manager.py`
- `scripts/webrtc_tracks.py`
- `scripts/realtime_inference.py`
- `scripts/inference.py`
- `scripts/preprocess.py`
- `scripts/run_api_server.sh`
- `scripts/run_turnserver.sh`
- `scripts/install_webrtc_deps.sh`

These are either alternate delivery paths, older/offline code, preparation utilities, or deployment helpers.

## Model Directory Findings

### `models/`

The root `models/` directory is mostly weights and config files, not hot-path source code.

Examples:

- `models/musetalk/musetalk.json`
- `models/musetalkV15/musetalk.json`
- `models/whisper/*`
- `models/sd-vae/*`
- `models/syncnet/*`

That means the actual optimization work belongs in code under `musetalk/`, not in the root `models/` directory.

### `musetalk/models/`

- `musetalk/models/unet.py`
  - Loads a `diffusers.UNet2DConditionModel`.
  - Does not explicitly configure an optimized attention processor.
- `musetalk/models/vae.py`
  - Decodes on GPU, then immediately converts to CPU NumPy/BGR in `decode_latents()`.
- `musetalk/models/syncnet.py`
  - Training/evaluation code, not on the live HLS inference path.

### `musetalk/utils/`

- `musetalk/utils/audio_processor.py`
  - One of the clearest remaining model-path optimization targets.
  - Encodes Whisper features on GPU, then moves them back to CPU.
  - Rebuilds frame-level prompts in a Python loop.
- `musetalk/utils/utils.py`
  - `datagen()` is used in direct streaming paths, but the shared HLS scheduler has its own batching path.
- `musetalk/utils/blending.py`
  - Compose/blending is CPU NumPy/OpenCV, not GPU-native.
- `musetalk/utils/face_parsing/*`
- `musetalk/utils/face_detection/*`
- `musetalk/utils/dwpose/*`
  - Relevant for avatar preparation, not the live throughput bottleneck once materials are prepared.

### `musetalk/whisper/`

- `musetalk/whisper/whisper/model.py`
  - Custom Whisper implementation uses a manual attention path.
  - However, the current live HLS runtime is using Hugging Face `WhisperModel`, not this implementation.
- `musetalk/whisper/audio2feature.py`
  - Legacy/alternate audio feature path, not the main active runtime path today.

## Current Hot Bottlenecks

### 1. Audio Prompt Construction Is Still Too CPU-Centric

In `musetalk/utils/audio_processor.py`:

- `encode_whisper_feature()` runs Whisper on GPU
- then returns `whisper_feature.detach().cpu().contiguous()`
- `build_audio_prompts()` then loops frame-by-frame in Python to construct prompts

This is one of the clearest remaining optimization opportunities because it adds:

- GPU to CPU transfer
- Python loop overhead
- extra CPU tensor manipulation before the scheduler even starts using the prompts

### 2. Scheduler Batch Assembly Still Pays Repeated CPU Staging Cost

In `scripts/hls_gpu_scheduler.py`:

- conditioning slices are copied into CPU staging buffers
- latent slices are gathered on CPU
- the padded batch is then copied to GPU

Even though batching is working well, there is still repeated host-side assembly around every GPU turn.

### 3. VAE Output Boundary Still Forces A CPU Round Trip

In `musetalk/models/vae.py`:

- `decode_latents_tensor()` produces a tensor on device
- `decode_latents()` immediately does:
  - `.detach().cpu()`
  - `.permute(...)`
  - `.numpy()`
  - RGB to BGR conversion

That means we are still paying a full host handoff right after decode.

### 4. Compose Is Still CPU/OpenCV

In `scripts/api_avatar.py` and `musetalk/utils/blending.py`:

- every decoded face is resized on CPU
- every output frame copies the base avatar frame
- mask blending is done in NumPy/OpenCV

This is not strictly “model math,” but it is still one of the largest remaining non-encode costs after the model step.

### 5. Encode Still Spawns A Fresh `ffmpeg` Per Chunk

In `scripts/api_avatar.py`:

- `_create_chunk()` launches a new subprocess for each chunk
- frames are streamed via stdin
- the chunk is encoded and muxed from scratch

This is still expensive, but it is more of a pipeline/encode bottleneck than a pure model/GPU bottleneck.

## Runtime Capability Findings

The current runtime supports:

- `torch 2.0.1+cu117`
- CUDA 11.7
- SDPA availability in PyTorch
- `diffusers 0.30.2`
- `transformers 4.39.2`
- `triton 2.0.0`

The current runtime does **not** currently have:

- `xformers`
- `flash_attn`

So the realistic next GPU-side attention optimization is:

- explicit SDPA-based optimization

not:

- xFormers-first
- FlashAttention-first

unless those dependencies are intentionally added later.

## Priority List

### Priority 1: Keep Conditioning Chunks On GPU

Why this is first:

- conditioning is currently built on CPU and copied to GPU every turn
- per-request conditioning memory is modest enough to make GPU residency realistic
- it directly reduces repeated H2D overhead in the hot path

Target files:

- `scripts/hls_gpu_scheduler.py`
- `musetalk/utils/audio_processor.py`
- `scripts/avatar_manager_parallel.py`

### Priority 2: Keep Avatar Latent Cycles On GPU

Why this is next:

- the scheduler currently gathers latents from CPU-resident avatar tensors
- latent cycles are relatively small compared with total available VRAM
- this can remove a repeated gather/stage/copy step from every generation turn

Target files:

- `scripts/api_avatar.py`
- `scripts/hls_gpu_scheduler.py`

### Priority 3: Vectorize Audio Prompt Building

Why this matters:

- `build_audio_prompts()` is still a Python loop over frames
- this is exactly the kind of overhead that scales poorly as concurrency rises
- it should become a tensorized sliding-window/gather operation

Target file:

- `musetalk/utils/audio_processor.py`

### Priority 4: Explicitly Optimize The UNet Attention Path

Why this matters:

- the runtime supports SDPA
- the UNet loader does not explicitly force the best available attention processor
- this is the cleanest model-level optimization after data residency improvements

Target files:

- `musetalk/models/unet.py`
- `scripts/avatar_manager_parallel.py`

### Priority 5: Rework The VAE Output Boundary

Why this matters:

- current decode returns CPU NumPy immediately
- some output formatting could happen before the transfer
- a larger future step would keep decoded ROIs on GPU long enough for GPU compose

Target files:

- `musetalk/models/vae.py`
- `scripts/hls_gpu_scheduler.py`
- `scripts/api_avatar.py`

### Priority 6: GPU Compose

Why this is later:

- it could be a meaningful win
- but it is riskier because it touches visual correctness
- it should come after the lower-risk data-residency improvements

Target files:

- `scripts/api_avatar.py`
- `musetalk/utils/blending.py`

## What Not To Prioritize

Based on the current code and prior experiments, these are lower-value next steps:

- pushing `HLS_SCHEDULER_MAX_BATCH` above `48`
- more worker-count tuning
- more startup-slice tuning
- changing `scripts/session_manager.py`
- changing WebRTC files for this HLS mission
- changing `musetalk/models/syncnet.py`
- optimizing the custom Whisper implementation before confirming the live runtime should even use it

## Practical Recommendation

If we want the strongest remaining model/GPU-focused branch to try, the best sequence is:

1. GPU-resident `conditioning_chunks`
2. GPU-resident avatar latent cycles
3. tensorized `build_audio_prompts()`
4. explicit SDPA attention optimization

That is the highest-confidence model/GPU optimization path left in the current codebase.

## Notes

- The startup-fairness scheduler logic in `scripts/hls_gpu_scheduler.py` should still be preserved.
- The current findings do **not** say encode/publish overhead is solved; only that the best remaining *model/GPU* gains are elsewhere first.
- This file is intended to be the reference document for the next implementation phase.
