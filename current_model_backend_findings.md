# MuseTalk Model And GPU Optimization Findings

## Scope

This document captures the current findings from a repo-wide throughput review focused on:

- the active HLS inference path
- the `scripts/` runtime code
- the inference-side model code under `musetalk/`
- the root entrypoints that actually drive live streaming

It is separate from `current_model_backend_acceleration_plan.md`, which is a broader implementation-planning document.

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

## Recent Experiment Updates

The direct GPU-resident conditioning experiment was tested in the shared HLS scheduler and then reverted.

Observed result at `concurrency=8`, `batch_size=4`, `playback_fps=24`, `musetalk_fps=12`:

- steady-state throughput stayed in the familiar band at about `1.961-1.965s` average segment interval
- tail latency stayed in the familiar band at about `3.08-3.21s` max segment interval
- startup fairness got worse:
  - one stream came live at about `3.0s`
  - the rest clustered around `5.0s`
  - `avg_time_to_live_ready_s` regressed to about `4.78s`

Conclusion:

- GPU-resident conditioning did **not** create a meaningful throughput gain
- it did **hurt** the startup behavior that the startup-first scheduler path had improved
- it should not remain the first optimization priority in the current branch

The later vectorized audio-prompt experiment was also tested and then rolled back.

Observed result at `concurrency=8`, `batch_size=4`, `playback_fps=24`, `musetalk_fps=12`:

- `avg_segment_interval_s = 2.0`
- `max_segment_interval_s = 3.225`
- `avg_time_to_live_ready_s = 4.772`

Conclusion:

- vectorizing `build_audio_prompts()` was a valid cleanup and correctness-preserving refactor
- but it did **not** materially improve throughput or startup behavior for the current HLS mission
- it should not remain the current first-priority performance bet in this branch either

The later GPU-resident latent-cycle experiment was also tested and then rolled back.

Observed result at `concurrency=8`, `batch_size=4`, `playback_fps=24`, `musetalk_fps=12`:

- run 1:
  - `avg_segment_interval_s = 1.971`
  - `max_segment_interval_s = 3.143`
  - `avg_time_to_live_ready_s = 4.774`
- run 2:
  - `avg_segment_interval_s = 1.99`
  - `max_segment_interval_s = 3.119`
  - `avg_time_to_live_ready_s = 4.774`

Conclusion:

- keeping avatar latent cycles on GPU did **not** produce a meaningful throughput shift
- it also did **not** repair the startup wave
- repeated latent gather/staging cost is therefore not the dominant limiter by itself
- this experiment is now rolled back and should not remain the current top priority either

The later explicit SDPA attention-path experiment was also tested and then rolled back.

Observed result across repeated `concurrency=8`, `batch_size=4`, `playback_fps=24`, `musetalk_fps=12` runs:

- run 1:
  - `avg_segment_interval_s = 1.971`
  - `max_segment_interval_s = 3.143`
  - `avg_time_to_live_ready_s = 4.774`
- run 2:
  - `avg_segment_interval_s = 1.99`
  - `max_segment_interval_s = 3.119`
  - `avg_time_to_live_ready_s = 4.774`
- later confirmation runs after the user double-checked the same stable start params:
  - `avg_segment_interval_s = 2.003`, `max_segment_interval_s = 3.222`, `avg_time_to_live_ready_s = 4.41`
  - `avg_segment_interval_s = 2.039`, `max_segment_interval_s = 3.104`, `avg_time_to_live_ready_s = 4.147`

Conclusion:

- explicit SDPA attention-path tuning did **not** materially improve throughput
- it also did **not** restore the better startup clustering
- this experiment is now rolled back and should not remain the current top priority either

We now also have a clean baseline benchmark for the isolated model path from `scripts/benchmark_pipeline.py`.

Observed result:

- best throughput: `51.0 fps` at `batch_size=16`
- max sustainable fps per stream at `8` concurrent: `6.4 fps`

Key timing breakdown from that benchmark:

- `batch_size=16`
  - `PE = 0.02 ms`
  - `UNet = 63.09 ms`
  - `VAE Full = 250.91 ms`
- `batch_size=48`
  - `PE = 0.02 ms`
  - `UNet = 186.41 ms`
  - `VAE Full = 764.67 ms`

Conclusion:

- the current PyTorch model path alone cannot reach the `96 fps` needed for `8 x 12 fps`
- the model/backend path is therefore a hard ceiling, not just an HLS delivery problem
- VAE decode is the dominant model-side bottleneck by a wide margin
- the backend acceleration priority should now start with VAE, then UNet

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
However, the direct GPU-latent-residency experiment did not materially move end-to-end throughput, so this staging cost is probably not the dominant limiter by itself.

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

So the realistic next *small* GPU-side attention optimization was:

- explicit SDPA-based optimization

That experiment has now been tested and rolled back, so the next serious path is no longer another small attention tweak. The remaining likely leverage is now in backend-level acceleration rather than another incremental PyTorch-path adjustment.

## Priority List

### Priority 1: TensorRT For VAE

Why this is first:

- the benchmark is now done
- VAE decode is the dominant model-side cost by a wide margin
- this is the highest-upside backend acceleration branch left

Target files:

- `musetalk/models/vae.py`
- `scripts/avatar_manager_parallel.py`
- `scripts/tensorrt_export.py`
- `scripts/trt_runtime.py`

### Priority 2: TensorRT For UNet

Why this is next:

- UNet is still a meaningful part of the model path
- but the benchmark showed it is materially smaller than VAE
- it should follow the VAE branch, not lead it

Target files:

- `musetalk/models/unet.py`
- `scripts/avatar_manager_parallel.py`
- `scripts/tensorrt_export.py`
- `scripts/trt_runtime.py`

### Priority 3: ONNX Runtime Fallback

Why this matters:

- this is the backup backend path if TensorRT is blocked
- it still has more remaining upside than returning to the smaller rolled-back PyTorch-path tweaks

Target files:

- `scripts/onnx_export.py`
- `scripts/ort_runtime.py`
- `scripts/avatar_manager_parallel.py`
- `scripts/hls_gpu_scheduler.py`

### Priority 4: Rework The VAE Output Boundary Only If Backend Acceleration Still Leaves A Gap

Why this is later:

- this is still a legitimate model-path target
- but the smaller PyTorch-path experiments have consistently failed to move the mission enough
- it is now better treated as a later follow-on optimization, not the lead branch

Target files:

- `musetalk/models/vae.py`
- `scripts/hls_gpu_scheduler.py`
- `scripts/api_avatar.py`

### Priority 5: GPU Compose Only If Backend Acceleration Still Leaves A Gap

Why this is later:

- it could still help
- but it remains a larger correctness-risk branch
- it should stay behind the cleaner backend-acceleration path

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

1. TensorRT for VAE
2. TensorRT for UNet
3. ONNX Runtime fallback if TensorRT is blocked
4. only after that, revisit VAE-output or GPU-compose follow-on work

That is the highest-confidence remaining model/GPU acceleration path left in the current codebase.

## Notes

- The startup-fairness scheduler logic in `scripts/hls_gpu_scheduler.py` should still be preserved.
- The current findings do **not** say encode/publish overhead is solved; only that the best remaining *model/GPU* gains are elsewhere first.
- Vectorized audio-prompt building was also a useful experiment, but it did not produce a meaningful throughput shift for the current 8-stream HLS target and should not be treated as an active win.
- GPU-resident latent cycles were also a useful experiment, but they did not produce a meaningful throughput shift for the current 8-stream HLS target and should not be treated as an active win either.
- GPU-resident conditioning was a useful experiment, but it is now a documented dead end for the current branch unless later evidence gives us a stronger reason to revisit it.
- Explicit SDPA attention tuning was also a useful experiment, but it did not produce a meaningful throughput shift for the current 8-stream HLS target and is now rolled back as well.
- The isolated model-path benchmark was the key turning point: it showed that the current PyTorch path tops out around `51 fps`, which means the backend/model path itself must change before the HLS system can ever reach the `96 fps` mission target.
- This file is intended to be the reference document for the next implementation phase.
