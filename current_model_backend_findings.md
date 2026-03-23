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

- best throughput: `51.1 fps` at `batch_size=32`
- max sustainable fps per stream at `8` concurrent: `6.4 fps`

Key timing breakdown from that broad baseline benchmark:

- `batch_size=32`
  - `PE = 0.02 ms`
  - `UNet = 119.92 ms`
  - `VAE Full = 505.75 ms`
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

Current backend-branch state:

- the repo now has a VAE backend hook, a TensorRT runtime loader, a TensorRT export script, and a repo-local TensorRT import shim
- the old current-environment TensorRT branch is still blocked and should remain untrusted
- the new alternate environment is now technically validated:
  - `/content/py310_trt_exp`
  - `torch==2.5.1`
  - `torch-tensorrt==2.5.0`
  - `tensorrt==10.3.0`
- a first successful alternate-env VAE engine artifact exists at:
  - `models/tensorrt_altenv/vae_decoder_trt.ts`
  - `models/tensorrt_altenv/vae_decoder_trt_meta.json`
  - batch range: `[1, 8]`
  - opt batch: `4`
  - saved size: about `137.5 MB`
- a newer broad-batch alternate-env VAE engine artifact now also exists at:
  - `models/tensorrt_altenv_bs32/vae_decoder_trt.ts`
  - `models/tensorrt_altenv_bs32/vae_decoder_trt_meta.json`
  - batch range: `[4, 48]`
  - opt batch: `16`
  - saved size: about `132 MB`
- runtime loading has also been validated in `/content/py310_trt_exp`
  with fallback disabled
- a broad-batch isolated benchmark now exists in the alternate env:
  - `benchmark_pipeline_trt_vae_bs32.json`
  - batch set: `[4, 8, 16, 32, 48]`
  - best throughput: `61.3 fps` at `batch_size=32`
- broad PyTorch comparison already exists:
  - `benchmark_pipeline_results.json`
  - best throughput: `51.1 fps` at `batch_size=32`
- practical meaning:
  - the TRT VAE backend is now materially faster across the broad `4..48`
    range
  - the isolated gain is about `+16.6%` to `+20.2%`
  - but the branch is still below the `96 fps` target for `8 x 12 fps`
- a backend-active HLS `api_server.py` startup is now validated in
  `/content/py310_trt_exp`
  - startup logs confirm `VAE decode backend active: tensorrt`
  - HLS scheduler started with `max_combined_batch_size=48`
- a backend-active HLS `load_test.py` result now also exists:
  - `load_test_report.json`
  - server shape:
    - eager UNet
    - TRT VAE
  - client settings:
    - `concurrency=1`
    - `batch_size=4`
    - `hold_seconds=10`
  - measured result:
    - `completed=1`
    - `failed=0`
    - `avg_time_to_live_ready_s=3.015`
    - `avg_segment_interval_s=0.196`
    - `max_segment_interval_s=1.512`

Important runtime caveats from that server validation:

- compiled UNet + TRT VAE is **not** the current validated HLS runtime shape
- a server run with `MUSETALK_COMPILE=1` and `MUSETALK_COMPILE_UNET=1`
  reached startup, but the first HLS generation batch failed during CUDA graph
  capture with:
  - `CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED`
  - `CUDA error: operation failed due to a previous error during capture`
- the current validated full-pipeline runtime shape is:
  - eager UNet
  - TRT VAE
- the end-to-end HLS test also revealed a new operational blocker:
  - ffmpeg repeatedly failed to open `h264_nvenc`
  - segment encode fell back to CPU `libx264`
  - observed errors included:
    - `OpenEncodeSessionEx failed: unsupported device (2)`
    - `No capable devices found`
    - repeated `Broken pipe` retries

Practical meaning:

- the TRT VAE branch is now validated past export/runtime/benchmark work and
  into real HLS output generation
- but higher-concurrency throughput claims are still confounded by the current
  NVENC failure / CPU-encode fallback path

## Backend Refactor Work Completed

The following repo-side changes are now in place for the VAE-first backend branch:

- `scripts/benchmark_pipeline.py`
  - created as the isolated model-path benchmark
  - later updated to understand backend-aware VAE decode
  - later updated to stop cleanly on `Ctrl+C`
- `musetalk/models/vae.py`
  - backend hook added for alternate VAE decode backends
  - original PyTorch decode path preserved as the fallback path
- `scripts/avatar_manager_parallel.py`
  - startup/runtime wiring added so the manager can attach a VAE backend when one exists
  - VAE compile is skipped when an accelerated backend is active
- `scripts/trt_runtime.py`
  - added as the runtime loader for a pre-exported TensorRT VAE decoder
- `scripts/tensorrt_export.py`
  - added as the export/build script for TensorRT backend artifacts
  - later patched to be compatible with the locally installed `torch-tensorrt==1.4.0`
- `scripts/api_avatar.py`
  - startup path patched to lazy-load preprocessing only when avatar
    preparation is actually requested
  - this allows the TRT HLS server to boot for existing prepared avatars
    without forcing `mmpose` imports at startup
- `tensorrt.py`
  - added at repo root as a compatibility shim because the NVIDIA wheel installs `tensorrt_bindings`, while `torch_tensorrt` imports `tensorrt`

This means the **repo integration work is materially done** for the VAE-first TensorRT branch,
and the alternate env now provides a real HLS server startup path for TRT VAE
validation.

## Current TensorRT Installation State

Stable environment `/content/py310`:

- Python `3.10`
- `torch 2.0.1+cu117`
- `torch_tensorrt` import currently fails
- `tensorrt` import currently fails
- practical meaning:
  - keep this env on the stable PyTorch server path for now
  - do **not** expect TRT-backed `api_server.py` runs to work here yet

Alternate TensorRT experiment environment `/content/py310_trt_exp`:

- Python `3.10`
- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchaudio==2.5.1`
- `torch-tensorrt==2.5.0`
- `tensorrt==10.3.0`
- additional currently installed export/runtime dependencies include:
  - `opencv-python`
  - `diffusers==0.30.2`
  - `transformers==4.39.2`
  - `accelerate==0.28.0`
  - `huggingface_hub==0.30.2`
  - `einops==0.8.1`
  - `safetensors==0.7.0`
  - `pillow==11.3.0`
  - `numpy==1.23.5`
- additional currently validated HLS/api_server dependencies include:
  - `fastapi==0.135.1`
  - `uvicorn==0.42.0`
  - `aiohttp==3.13.3`
  - `soundfile==0.12.1`
  - `librosa==0.11.0`
  - `imageio==2.37.3`
  - `omegaconf==2.3.0`
  - `ffmpeg-python==0.2.0`
  - `aiofiles==24.1.0`
  - `av==17.0.0`

Important packaging lesson from the server-validation step:

- keep `numpy==1.23.5` in this env
- do **not** install unpinned `gradio` into the TRT env for HLS validation
- a previous drift to:
  - `numpy 2.2.6`
  - `gradio 6.9.0`
  - `huggingface_hub 1.7.2`
  broke OpenCV imports and the `transformers` / `tokenizers` compatibility
  family

Important clarification:

- the TensorFlow `TF-TRT Warning: Could not find TensorRT` lines that appear in logs are still noise from TensorFlow imports
- they are **not** the current blocker for our MuseTalk scripts

## Export History And Current Status

Current historical interpretation:

- the original `torch 2.0.1 + torch-tensorrt 1.4.0 + TensorRT 8.6.1` branch
  reached repo integration but not a trustworthy runtime path
- an earlier engine under `models/tensorrt/` should still be treated as stale
  and untrusted for performance validation

What changed in the alternate env:

- the exporter was patched to normalize old `dynamo_compile` references to
  `dynamo`
- the exporter save path was patched so `torch_tensorrt.save(...)` receives
  example inputs when the compiled result is a `torch.fx.GraphModule`
- the first successful alternate-env export used:
  - `--components vae`
  - `--batch-sizes 1,4,8`
  - output dir: `models/tensorrt_altenv`
- a later broader export then used:
  - `--components vae`
  - `--batch-sizes 4,8,16,32,48`
  - output dir: `models/tensorrt_altenv_bs32`

Current active export result:

- saved artifact:
  - `models/tensorrt_altenv_bs32/vae_decoder_trt.ts`
- metadata artifact:
  - `models/tensorrt_altenv_bs32/vae_decoder_trt_meta.json`
- final metadata:
  - batch range `[4, 48]`
  - opt batch `16`
  - output shape `[3, 256, 256]`
- final save line reported:
  - about `132 MB`
- final export log reported:
  - `Compilation finished ... using ir=dynamo`
  - `TensorRT export complete`

Runtime validation after export:

- `scripts.trt_runtime.load_vae_trt_decoder(...)` successfully loaded the
  alternate-env engine with fallback disabled
- a direct decode test succeeded on `cuda:0`
- output shape matched expectations:
  - `(B, 3, 256, 256)`
- a smoke benchmark also succeeded and explicitly logged:
  - `VAE decode backend: tensorrt`

Current practical conclusion:

- the VAE TensorRT branch is now **technically valid** in the alternate env
  for:
  - export
  - runtime load
  - decode execution
  - isolated benchmark activation
- the branch is now **performance-validated for the isolated model path only**
  - broad best-throughput comparison:
    - `51.1 fps` PyTorch
    - `61.3 fps` TRT VAE
  - broad VAE full-path reduction:
    - `batch_size=16`: `253.34 -> 197.75 ms`
    - `batch_size=32`: `505.75 -> 396.20 ms`
    - `batch_size=48`: `764.93 -> 596.55 ms`
- the branch is still **not sufficient for the end goal**
  - the measured ceiling is still only about `7.7 fps/stream` at `8`
    concurrent from the model path alone
- the branch is now **server-reproducible for first HLS output**
  - `/content/py310` still cannot use TRT
  - `/content/py310_trt_exp` now starts the HLS server and has produced a
    successful backend-active single-stream HLS run
  - higher-concurrency HLS validation is still pending, and current results are
    still confounded by `h264_nvenc` failure / `libx264` fallback
  - that milestone is now superseded by the visual-correctness failure recorded
    below

## Critical Correctness Failure In The Active TRT Artifact

The current active broad-batch TensorRT VAE artifact is now confirmed to be
**visually wrong** and should be treated as **untrusted** for lip-sync
validation, quality review, or production demos.

Observed regression:

- HLS `/wall` output shows the talking-face ROI covered by a flat gray mask
- the regression appeared after switching the face decode path to the exported
  TensorRT VAE in `models/tensorrt_altenv_bs32`

What was inspected across the active runtime path:

- `scripts/avatar_manager_parallel.py`
  - attaches the TRT backend to `musetalk/models/vae.py`
- `scripts/hls_gpu_scheduler.py`
  - runs UNet, then calls `self.manager.vae.decode_latents(...)`
- `scripts/api_avatar.py`
  - resizes the decoded face patch and calls `compose_frame(...)`
- `musetalk/utils/blending.py`
  - only blends the already-decoded face ROI back into the avatar frame
- prepared avatar assets under `results/v15/avatars/test_avatar`
  - full frames, masks, coords, and cached materials are present

What the direct A/B checks showed:

- cached/avatar latent decode:
  - PyTorch face output: range `0..250`, mean `63.2`
  - TRT face output: range `104..136`, mean `120.2`
  - mean absolute error: about `87.1`
- real UNet-predicted latent decode:
  - PyTorch face output: range `0..236`, mean `116.6`
  - TRT face output: range `93..150`, mean `120.3`
  - mean absolute error: about `53.3`
- wrapper control test in pure PyTorch:
  - `scripts/tensorrt_export.py` wrapper matches the normal PyTorch VAE path
  - mean absolute error: about `5.3e-05`
  - max absolute error: about `0.0022`

Practical conclusion:

- the gray-mask regression is **not** a compositor or mask-asset problem
- the active TRT engine is already returning a collapsed mid-gray face tensor
  before blend
- the current evidence is now stronger than that:
  - the fault is already present in the **in-memory compiled TRT module**
  - it is therefore not primarily a save/load-only regression
- current TRT throughput wins should therefore be treated as **performance-only
  measurements**, not as proof of a production-valid backend
- post-patch direct validation of the active broad-batch artifact still fails:
  - `scripts/validate_vae_backend.py`
  - PyTorch output range: `0.0..0.9814`, mean `0.2485`
  - TRT output range: `0.3989..0.5337`, mean `0.4714`
  - MAE: about `0.3408`
- a newer exact-batch FP16 artifact also now fails validation with almost the
  same error signature:
  - artifact dir: `models/tensorrt_fp16_bs4`
  - batch range: `[4, 4]`
  - save format: `torchscript`
  - validation MAE: about `0.340751`
  - TRT output range: `0.3989..0.5322`, mean `0.4714`
- this means the regression is not explained by the earlier broad dynamic
  `[4..48]` profile by itself
- the current strongest hypothesis is now:
  - the FP16 TRT VAE compile/runtime behavior is numerically wrong on this
    stack
  - dynamic shapes may still worsen things, but they are no longer the main
    explanation by themselves
- a later direct in-memory compile check now confirms that:
  - `scripts/validate_vae_trt_inmemory.py`
  - batch size: `4`
  - precision: `fp16`
  - output range from the in-memory TRT module: `0.3989..0.5342`, mean
    `0.4714`
  - MAE vs PyTorch: `0.3407516`
  - practical meaning:
    - the corruption is already present before any artifact serialization or
      reload
- a later stage-by-stage decoder inspection also now localizes the earliest
  bad region:
  - `scripts/inspect_vae_trt_stages.py`
  - first bad stage: `decoder_mid_block`
  - exact-match stages before divergence:
    - `scale_post_quant`
    - `decoder_conv_in`
  - first divergent stage metrics:
    - `decoder_mid_block`
    - MAE: `0.4712`
    - max abs: `6.0391`
  - later stages continue to drift and amplify the error
  - `output_normalize` matches exactly when fed the same pre-normalized tensor,
    which means the final clamp/normalize step is **not** the culprit

New TRT-tooling guardrails now added:

- `scripts/validate_vae_backend.py` is now reusable as a backend-comparison
  helper instead of only a one-off CLI script
- `scripts/validate_vae_trt_inmemory.py` can compare PyTorch vs TRT output
  before any save/load boundary
- `scripts/inspect_vae_trt_stages.py` can localize the first divergent decoder
  stage under TRT
- `scripts/tensorrt_export.py` now supports post-export VAE correctness
  validation and writes validation metadata into `vae_decoder_trt_meta.json`
- `scripts/trt_runtime.py` now supports
  `MUSETALK_TRT_REQUIRE_VALIDATION=1` so a caller can refuse artifacts that are
  missing validation metadata or explicitly marked invalid

Additional serialization/runtime findings from the newer TRT stack:

- the old loadable broad-batch artifact in `models/tensorrt_altenv_bs32`
  remains visually wrong
- the newer `exported_program` save attempt in
  `models/tensorrt_altenv_ep_bs32_v3` saved successfully but later failed
  reload with symbolic-shape deserialization errors (`KeyError: s0`)
- the retraced `exported_program` attempt in
  `models/tensorrt_altenv_ep_bs32_v4` failed during save because the TRT engine
  torchbind object tripped `state_dict()` / `get_extra_state()` with:
  - `NotImplementedError: '__len__' is not implemented for __torch__.torch.classes.tensorrt.Engine`
- a later exact-batch FP16 retry using `save_format=torchscript` in
  `models/tensorrt_fp16_bs4` got past compile and save cleanly, then failed
  the new post-export validation gate
- taken together, the current `torch==2.5.1` + `torch-tensorrt==2.5.0` VAE
  serialization/runtime path still appears untrustworthy for a correctness-safe
  MuseTalk VAE decode backend

Current best next-step investigation order:

1. keep the original monolithic TRT artifact path marked broken/untrusted
2. continue on the new **stagewise TRT VAE** branch in `scripts/trt_runtime.py`
3. treat `batch_size=4` as the first repaired bucket, not the finish line
4. validate stagewise correctness bucket-by-bucket (`8`, `16`, `32`, `48`)
5. benchmark stagewise throughput only after the wider buckets stay correct
6. then widen HLS `/wall` and load testing once stagewise TRT stays both
   correct and faster than the PyTorch VAE path

New stagewise backend findings:

- the repo now has an experimental backend named `trt_stagewise`
- this backend compiles exact-batch decoder stages independently and caches
  them by batch size at runtime
- it keeps `native_group_norm` on the PyTorch side during stage compilation
- the first end-to-end correctness result is now strong:
  - script:
    - `scripts/validate_vae_backend.py`
  - backend:
    - `trt_stagewise`
  - batch size:
    - `4`
  - report:
    - `tmp/vae_stagewise_backend_validation_bs4/report.json`
  - metrics:
    - output min/max/mean now track PyTorch closely
    - MAE: `0.0005082`
    - max abs: `0.0097656`
- stage-local probes with the same fallback now show:
  - `decoder_mid_block`: repaired, MAE `0.00419`
  - `decoder_up_block_0`: passes, MAE `0.01746`
  - `decoder_postprocess`: passes, MAE `0.000489`
  - `decoder_up_block_1` and `decoder_up_block_2` still drift slightly when
    isolated, but the full stagewise `batch_size=4` decode is now visually and
    numerically close to PyTorch

Current interpretation:

- the gray-mask problem is not fixed by the old saved-engine TRT route
- it **is** effectively fixed at `batch_size=4` by the new stagewise TRT path
- a later real HLS `/wall` check now also appears visually correct on the
  stagewise path at the same `batch_size=4` setup
- that is the first user-visible confirmation that the repair is not only
  synthetic or metric-based
- the next risk is no longer “is it gray?” but:
  - does stagewise TRT remain correct at larger batches?
  - and is it still fast enough to justify runtime adoption?

## Separate TensorRT Environment Attempt Update

The alternate-env setup branch is no longer theoretical.

Current confirmed state:

- `/content/py310_trt_exp` exists and imports the pinned backend family cleanly
- the active broad-batch VAE export now exists in `models/tensorrt_altenv_bs32`
- runtime load is validated there with fallback disabled
- the GPU machine no longer shows the earlier disk-pressure blocker that killed
  the first setup attempt

Important packaging lesson that still stands:

- keep the backend stack pinned
- do **not** use unpinned `pip install torch-tensorrt tensorrt`

Current next step from the alternate env branch:

- run HLS load testing there with the broad-batch TRT artifact
- use the successful HLS server startup as the entry point, not the stable
  `/content/py310` server
- if that still leaves a large gap, the next meaningful branch is likely UNet
  acceleration after server validation

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
- A later March 22, 2026 benchmark on the slower Threadripper server reproduced the same model tier at about `51.1 fps`, which strongly suggests the cross-server performance gap is not in raw UNet/VAE inference. The difference is more likely in the host-side HLS path around compose, encode, queueing, and CPU/memory behavior.
- This file is intended to be the reference document for the next implementation phase.
