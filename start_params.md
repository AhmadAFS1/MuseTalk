# Start Params

This file is the current reference launch config for the stable `max_batch=48` HLS throughput test path.

## Server Start Command

```bash
unset PYTORCH_CUDA_ALLOC_CONF

export MUSETALK_COMPILE=1
export MUSETALK_COMPILE_UNET=1
export MUSETALK_COMPILE_VAE=1
export MUSETALK_COMPILE_MODE=reduce-overhead
export MUSETALK_COMPILE_TRACEBACK=1
export MUSETALK_COMPILE_WARMUP_BATCHES=4,8,16,32,48
export MUSETALK_WARM_RUNTIME=1

export AVATAR_CACHE_MAX_AVATARS=0
export AVATAR_CACHE_MAX_MEMORY_MB=12000
export AVATAR_CACHE_TTL_SECONDS=3600

export HLS_SCHEDULER_MAX_BATCH=48
export HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,32,48
export HLS_SCHEDULER_STARTUP_SLICE_SIZE=4
export HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS=999
export HLS_STARTUP_CHUNK_DURATION_SECONDS=0.5
export HLS_STARTUP_CHUNK_COUNT=1
export HLS_PREP_WORKERS=8
export HLS_COMPOSE_WORKERS=8
export HLS_ENCODE_WORKERS=8
export HLS_MAX_PENDING_JOBS=24
export HLS_CHUNK_VIDEO_ENCODER=h264_nvenc
export HLS_PERSISTENT_SEGMENTER=0

python api_server.py --host 0.0.0.0 --port 8000
```

## Threadripper Variant

This is the current best-known start block for the large-core Threadripper host investigated on March 22, 2026.

Why this variant exists:

- the shared HLS path is still partly CPU-bound on compose + encode
- the Threadripper box improved with a modest worker increase
- a more aggressive increase regressed startup and worst-case segment timing again

Current recommendation for that machine:

- prefer `HLS_PREP_WORKERS=12`
- prefer `HLS_COMPOSE_WORKERS=10`
- prefer `HLS_ENCODE_WORKERS=10`
- do **not** assume higher worker counts are always better

Copy/paste block:

```bash
unset PYTORCH_CUDA_ALLOC_CONF

export MUSETALK_COMPILE=1
export MUSETALK_COMPILE_UNET=1
export MUSETALK_COMPILE_VAE=1
export MUSETALK_COMPILE_MODE=reduce-overhead
export MUSETALK_COMPILE_TRACEBACK=1
export MUSETALK_COMPILE_WARMUP_BATCHES=4,8,16,32,48
export MUSETALK_WARM_RUNTIME=1

export AVATAR_CACHE_MAX_AVATARS=0
export AVATAR_CACHE_MAX_MEMORY_MB=12000
export AVATAR_CACHE_TTL_SECONDS=3600

export HLS_SCHEDULER_MAX_BATCH=48
export HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,32,48
export HLS_SCHEDULER_STARTUP_SLICE_SIZE=4
export HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS=999
export HLS_STARTUP_CHUNK_DURATION_SECONDS=0.5
export HLS_STARTUP_CHUNK_COUNT=1
export HLS_PREP_WORKERS=12
export HLS_COMPOSE_WORKERS=10
export HLS_ENCODE_WORKERS=10
export HLS_MAX_PENDING_JOBS=24
export HLS_CHUNK_VIDEO_ENCODER=h264_nvenc
export HLS_PERSISTENT_SEGMENTER=0

python api_server.py --host 0.0.0.0 --port 8000
```

Known caution from the same server:

- `HLS_PREP_WORKERS=16`
- `HLS_COMPOSE_WORKERS=12`
- `HLS_ENCODE_WORKERS=12`

looked more aggressive on paper, but regressed `avg_time_to_live_ready_s` and `max_segment_interval_s`.

## Full Param Reference

The full current explanation of what each launch param actually does now lives in:

- [`current_start_param_reference.md`](./current_start_param_reference.md)

## Matching Load Test Command

```bash
python load_test.py \
  --base-url http://localhost:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --batch-size 4 \
  --playback-fps 24 \
  --musetalk-fps 12 \
  --hold-seconds 120
```

## Baseline Benchmark Command

Run this before the TensorRT / ONNX branch.

Important:

- stop `api_server.py` first so the benchmark can own GPU memory
- this is a model-path benchmark, not a replacement for `load_test.py`

```bash
cd /content/MuseTalk
/content/py310/bin/python scripts/benchmark_pipeline.py \
  --batch-sizes 4,8,16,24,32,40,48 \
  --warmup 20 \
  --iters 50 \
  --output-json benchmark_pipeline_results.json
```

This writes:

- `benchmark_pipeline_results.json`

and prints the per-batch timings for:

- positional encoding
- UNet
- VAE decode
- GPU to CPU transfer
- total pipeline throughput

## Latest Stable PyTorch Baseline

Latest stable-env isolated model-path baseline from `scripts/benchmark_pipeline.py`:

- best throughput: `51.1 fps` at `batch_size=32`
- max sustainable fps per stream at `8` concurrent: `6.4 fps`

Most important interpretation of the stable PyTorch baseline:

- this is below the `96 fps` needed for `8 x 12 fps`
- so the current PyTorch model path is a hard ceiling before HLS compose/encode overhead is even added
- VAE decode is the dominant model-side bottleneck

That is why the next serious branch is now backend acceleration, with VAE first.

## Current Backend Branch State

Repo status today:

- keep the stable HLS server launch block above pointed at `/content/py310`
- `/content/py310` is still the stable PyTorch server env
- `/content/py310` is still **not** TRT-ready
  - `torch_tensorrt` import fails there
  - `tensorrt` import fails there
- the successful TRT work now lives in `/content/py310_trt_exp`
  - `torch==2.5.1`
  - `torch-tensorrt==2.5.0`
  - `tensorrt==10.3.0`
- the first successful TRT VAE artifact in the alternate env was:
  - `/content/MuseTalk/models/tensorrt_altenv/vae_decoder_trt.ts`
  - `/content/MuseTalk/models/tensorrt_altenv/vae_decoder_trt_meta.json`
  - batch range `[1, 8]`
  - opt batch `4`
- the newer broad-batch TRT VAE artifact is now the active validation one:
  - `/content/MuseTalk/models/tensorrt_altenv_bs32/vae_decoder_trt.ts`
  - `/content/MuseTalk/models/tensorrt_altenv_bs32/vae_decoder_trt_meta.json`
  - batch range `[4, 48]`
  - opt batch `16`
- runtime loading has been validated there with fallback disabled
- backend-active benchmarking has also been validated there

Practical meaning:

- VAE TensorRT conversion is now real
- the broad-batch TRT VAE engine is now materially faster than the old PyTorch
  baseline on the same `4..48` batch range
- but it is still only validated in the alternate env
- do **not** point the stable `/content/py310` server at the TRT artifact yet

## Current TensorRT Validation State

Current active broad-batch TRT benchmark:

- file:
  - `benchmark_pipeline_trt_vae_bs32.json`
- batch set:
  - `[4, 8, 16, 32, 48]`
- best throughput:
  - `61.3 fps` at `batch_size=32`
- implied model-path ceiling at `8` concurrent:
  - about `7.7 fps/stream`

Broad PyTorch comparison:

- file:
  - `benchmark_pipeline_results.json`
- best throughput:
  - `51.1 fps` at `batch_size=32`

Most important measured deltas:

- `batch_size=4`
  - throughput: `45.8 -> 53.4 fps`
  - VAE full path: `65.04 -> 51.36 ms`
- `batch_size=16`
  - throughput: `50.5 -> 60.2 fps`
  - VAE full path: `253.34 -> 197.75 ms`
- `batch_size=32`
  - throughput: `51.1 -> 61.3 fps`
  - VAE full path: `505.75 -> 396.20 ms`
- `batch_size=48`
  - throughput: `50.9 -> 61.2 fps`
  - VAE full path: `764.93 -> 596.55 ms`

Practical meaning:

- the broad-batch TRT VAE backend is now a **material** improvement
- the isolated model-path gain is about `+16.6%` to `+20.2%` across `4..48`
- but it is still far short of the `96 fps` target for `8 x 12 fps`
- this has now crossed the first HLS `api_server.py` startup milestone in
  `/content/py310_trt_exp`
- the first backend-active HLS `load_test.py` run has now completed
- current end-to-end validation is for **eager UNet + TRT VAE**
- it is **not** yet proof that the full HLS goal is solved

## Critical Correctness Update

The current active TRT VAE artifact is now **functionally broken** for visual
output and should be treated as **untrusted** for lip-sync validation or
production demos.

Observed regression:

- the talking-face ROI in HLS `/wall` output is replaced by a flat gray mask
- this happens after switching the VAE decode path to the exported TensorRT
  engine in `models/tensorrt_altenv_bs32`

What has now been validated:

- the problem is **not** the avatar masks or prepared avatar materials
- the problem is **not** `get_image_blending(...)` in
  `musetalk/utils/blending.py`
- the problem is **not** the wrapper math in
  `scripts/tensorrt_export.py` when run in pure PyTorch
- the current broken output is already present in the decoded face patch before
  it is blended back into the avatar frame

Representative A/B checks against the PyTorch VAE path:

- cached/avatar latent decode:
  - PyTorch face output: range `0..250`, mean `63.2`
  - TRT face output: range `104..136`, mean `120.2`
  - mean absolute error: about `87.1`
- real UNet-predicted latent decode:
  - PyTorch face output: range `0..236`, mean `116.6`
  - TRT face output: range `93..150`, mean `120.3`
  - mean absolute error: about `53.3`
- export-wrapper control test in pure PyTorch:
  - mean absolute error: about `5.3e-05`
  - max absolute error: about `0.0022`

Practical meaning:

- the `61.3 fps` TRT benchmark is still useful as a raw speed datapoint
- but the current TRT artifact is **not decision-grade**
- do **not** treat the current TRT branch as visually validated
- use the stable PyTorch VAE path for any real lip-sync or avatar-quality check
  until this correctness regression is fixed
- direct post-patch validation in `/content/py310_trt_exp` still fails against
  the current active broad-batch artifact:
  - `python scripts/validate_vae_backend.py --avatar-id test_avatar --trt-dir ./models/tensorrt_altenv_bs32`
  - PyTorch output range: `0.0..0.9814`, mean `0.2485`
  - TRT output range: `0.3989..0.5337`, mean `0.4714`
  - MAE: about `0.3408`
- a newer exact-batch FP16 export also now fails the same way:
  - artifact dir: `models/tensorrt_fp16_bs4`
  - export shape: batch `[4, 4]`
  - save path: `torchscript`
  - compile: passed
  - save: passed
  - validation: failed
  - MAE: about `0.340751`
- practical meaning:
  - the gray-mask bug is **not** just a broad dynamic `[4..48]` profile issue
  - the bug still reproduces with an exact `batch_size=4` FP16 TRT VAE
  - the current evidence now points more strongly at the FP16 TRT VAE
    compile/runtime behavior itself
- a later in-memory TRT compile check now confirms the fault is already present
  before save/load:
  - `python scripts/validate_vae_trt_inmemory.py --avatar-id test_avatar --batch-size 4 --precision fp16`
  - PyTorch output range: `0.0..0.9814`, mean `0.2485`
  - TRT in-memory output range: `0.3989..0.5342`, mean `0.4714`
  - MAE: about `0.3407516`
- a later stage-by-stage decoder check now localizes the first bad region:
  - `python scripts/inspect_vae_trt_stages.py --avatar-id test_avatar --batch-size 4 --precision fp16`
  - first bad stage: `decoder_mid_block`
  - `scale_post_quant` and `decoder_conv_in` still match exactly
  - `output_normalize` also matches exactly when given the same input
  - practical meaning:
    - the fault is in the decoder core, not the final output clamp/scale

Current TRT guardrails now added in code:

- `scripts/validate_vae_backend.py` is now reusable as a correctness checker,
  not just a one-off script
- `scripts/validate_vae_trt_inmemory.py` compares PyTorch vs TRT output before
  any save/load boundary
- `scripts/inspect_vae_trt_stages.py` localizes the first divergent decoder
  stage under TRT
- `scripts/tensorrt_export.py` now supports post-export VAE validation metadata
  with:
  - `--validate-avatar-id`
  - `--validate-batch-size`
  - `--validate-max-mae`
  - `--require-valid-vae`
- `scripts/trt_runtime.py` now supports:
  - `MUSETALK_TRT_REQUIRE_VALIDATION=1`
  - this refuses to activate artifacts that are missing validation metadata or
    are explicitly marked invalid

Recommended correctness gate before any future TRT HLS run:

```bash
cd /content/MuseTalk
source /content/py310_trt_exp/bin/activate
python scripts/validate_vae_backend.py \
  --avatar-id test_avatar \
  --trt-dir ./models/tensorrt_altenv_bs32 \
  --output-dir ./tmp/vae_backend_validation_current
```

## Current Next-Step Plan

The correctness-first branch has now produced the first TRT path that appears
to work visually on real HLS output.

Planned sequence:

1. Keep treating the monolithic TRT artifact path as broken.
   - the broad-batch artifact is still gray-mask broken
   - the exact `batch_size=4` FP16 artifact is also broken
2. Use the new stagewise TRT backend as the active repair branch.
   - backend name: `trt_stagewise`
   - exact-batch decoder stages compiled independently
   - `native_group_norm` kept on the PyTorch side inside each compiled stage
3. Validate stagewise correctness on real cached latents before wider HLS use.
   - `batch_size=4` is now validated
   - report: `tmp/vae_stagewise_backend_validation_bs4/report.json`
   - full-image MAE: `0.000508`
   - output range now matches PyTorch instead of collapsing to gray
4. The first HLS `/wall` visual check with `trt_stagewise` is now encouraging.
   - real mouth output is visible again instead of the gray ROI collapse
   - this was seen on the stagewise `batch_size=4` server configuration
5. Next expand that validation to larger scheduler buckets (`8`, `16`, `32`, `48`)
   and benchmark throughput before widening the live HLS config.
6. Only if stagewise TRT stays visually correct and still faster than PyTorch
   should it replace the PyTorch VAE path for the normal shared-batch HLS setup.

Target outcome:

- visual output that matches the PyTorch VAE path closely
- runtime still materially faster than the current PyTorch decode path
- only then resume end-to-end HLS throughput validation

## Stagewise TRT Update

The first real correctness fix is now in the repo:

- `scripts/trt_runtime.py`
  - added `trt_stagewise` VAE backend
  - compiles exact-batch decoder stages on demand and caches them by batch size
  - uses `native_group_norm` PyTorch fallback inside stagewise TRT compilation
- `scripts/inspect_vae_trt_stages.py`
  - stage-local correctness probes now show:
    - `decoder_mid_block`: fixed with `native_group_norm`, MAE `0.00419`
    - `decoder_up_block_0`: MAE `0.01746`
    - `decoder_postprocess`: MAE `0.000489`
- `scripts/validate_vae_backend.py`
  - now accepts `--backend`
  - stagewise validation at `batch_size=4` succeeded:
    - backend: `trt_stagewise`
    - MAE: `0.000508`
    - max abs: `0.00977`
    - report: `tmp/vae_stagewise_backend_validation_bs4/report.json`

Current interpretation:

- the old saved-engine TRT path is still untrusted for visuals
- the new stagewise in-memory TRT backend is the first path that looks
  PyTorch-correct enough to continue testing
- a later real `/wall` check now also appears visually good at the validated
  `batch_size=4` setup
- so the stagewise branch has crossed the first “looks right on the avatar”
  milestone, not just the synthetic decode-validation milestone

## Current Environment Split

Stable env `/content/py310`:

- keep using this for the stable HLS server path
- keep the launch block at the top of this file unchanged for now
- do **not** expect TRT-backed server runs to work here yet

Alternate env `/content/py310_trt_exp`:

- validated for:
  - export
  - runtime load
  - isolated benchmark
  - HLS `api_server.py` startup with TRT VAE active
- current HLS server startup is now validated there for existing prepared
  avatars
- WebRTC is still disabled there until `aiortc` is installed
- avatar preparation inside this env is still not the validated path
  - the live workaround is a lazy preprocessing import in
    `scripts/api_avatar.py`
  - that lets the server load already-prepared avatars without forcing the
    `mmpose` stack at startup

## Current Benchmark Commands

Re-run the active broad-batch TRT benchmark if needed:

```bash
cd /content/MuseTalk
source /content/py310_trt_exp/bin/activate
MUSETALK_TRT_ENABLED=1 \
MUSETALK_VAE_BACKEND=trt \
MUSETALK_TRT_FALLBACK=0 \
MUSETALK_TRT_DIR=/content/MuseTalk/models/tensorrt_altenv_bs32 \
MUSETALK_COMPILE_VAE=0 \
python scripts/benchmark_pipeline.py \
  --batch-sizes 4,8,16,32,48 \
  --warmup 5 \
  --iters 20 \
  --output-json benchmark_pipeline_trt_vae_bs32.json
```

Current recommendation:

- do **not** try to run TRT from `/content/py310`
- use `scripts/setup_trt_experiment_env.sh --install-server-deps` to recreate
  the validated TRT HLS env cleanly
- use the old broad-batch `trt` backend only for performance/debug comparisons
- use the new `trt_stagewise` backend for current visual HLS verification
- even with the improvement, expect more acceleration work to still be needed
  after that
- if you want runtime safety against visually broken TRT artifacts, add:

```bash
export MUSETALK_TRT_REQUIRE_VALIDATION=1
```

## Experimental TRT VAE `api_server.py` Start

This is the launch shape to use the exported TRT VAE artifact in the alternate
environment for **debugging and performance experiments only**.

Important current warning:

- the active broad-batch TRT VAE artifact currently produces a gray-mask face
  ROI in visual output
- this server block is therefore **not** the recommended path for lip-sync
  validation right now
- keep using the stable PyTorch VAE path for any visual-quality signoff

Current validated dependency rule:

- keep the TRT env on the pinned package family that now works for HLS:
  - `numpy==1.23.5`
  - `opencv-python==4.9.0.80`
  - `huggingface_hub==0.30.2`
- do **not** pull unpinned server/UI packages into the TRT env
  - the earlier drift to `numpy 2.2.6`, `gradio 6.9.0`, and
    `huggingface_hub 1.7.2` broke OpenCV imports and the `transformers`
    dependency family
- `scripts/setup_trt_experiment_env.sh --install-server-deps` now installs the
  validated HLS/api_server dependency set without bringing in `gradio` or
  `moviepy`

Current validated startup result:

- `api_server.py` now starts successfully in `/content/py310_trt_exp`
- logs confirm:
  - `VAE decode backend active: tensorrt`
  - HLS scheduler started with `max_combined_batch_size=48`
- WebRTC still reports:
  - `WebRTC disabled (missing deps): No module named 'aiortc'`

Important runtime clarification:

- the current **validated** end-to-end HLS runtime shape is:
  - UNet in eager PyTorch
  - VAE decode in TensorRT
- a later server run with compiled UNet + TRT VAE did start successfully, but
  the first HLS generation batch failed inside CUDA graph capture with:
  - `CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED`
  - `CUDA error: operation failed due to a previous error during capture`
- for current HLS validation, keep UNet compile disabled

Important batching note:

- the active broad-batch TRT VAE engine now supports combined batch range
  `[4, 48]`
- that means this server block can now mirror the stable shared HLS batch
  buckets for a fairer TRT-vs-PyTorch validation
- the engine's current metadata is:
  - artifact dir: `/content/MuseTalk/models/tensorrt_altenv_bs32`
  - batch range: `[4, 48]`
  - opt batch: `16`

Copy/paste block:

```bash
cd /content/MuseTalk
source /content/py310_trt_exp/bin/activate

unset PYTORCH_CUDA_ALLOC_CONF

export MUSETALK_COMPILE=0
export MUSETALK_COMPILE_UNET=0
export MUSETALK_COMPILE_VAE=0
export MUSETALK_WARM_RUNTIME=1

export MUSETALK_TRT_ENABLED=1
export MUSETALK_VAE_BACKEND=trt
export MUSETALK_TRT_FALLBACK=0
export MUSETALK_TRT_DIR=/content/MuseTalk/models/tensorrt_altenv_bs32

export AVATAR_CACHE_MAX_AVATARS=0
export AVATAR_CACHE_MAX_MEMORY_MB=12000
export AVATAR_CACHE_TTL_SECONDS=3600

export HLS_SCHEDULER_MAX_BATCH=48
export HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,32,48
export HLS_SCHEDULER_STARTUP_SLICE_SIZE=4
export HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS=999
export HLS_STARTUP_CHUNK_DURATION_SECONDS=0.5
export HLS_STARTUP_CHUNK_COUNT=1
export HLS_PREP_WORKERS=8
export HLS_COMPOSE_WORKERS=8
export HLS_ENCODE_WORKERS=8
export HLS_MAX_PENDING_JOBS=24
export HLS_CHUNK_VIDEO_ENCODER=h264_nvenc
export HLS_PERSISTENT_SEGMENTER=0

python api_server.py --host 0.0.0.0 --port 8000
```

If the server starts correctly with TRT active, the startup logs should show:

- `VAE decode backend active: tensorrt`
- `torch.compile disabled (set MUSETALK_COMPILE=1 to enable)`
- scheduler settings consistent with:
  - `max_combined_batch_size=48`
  - `fixed_batch_sizes=[4, 8, 16, 32, 48]`

## Current Working Stagewise TRT `api_server.py` Start

This is the current best visual-validation startup shape for TRT in the
alternate env.

Current known-good characteristics:

- backend: `trt_stagewise`
- UNet: eager PyTorch
- VAE: stagewise TRT with `native_group_norm` kept on the PyTorch side
- scheduler: fixed `batch_size=4`
- current visual result:
  - real mouth motion appears again on HLS `/wall`
  - the earlier gray ROI collapse is not visible in the current wall check

Current important caveats:

- this is the visually repaired path, **not** the broad-batch throughput path
- it is currently validated at `batch_size=4`
- larger buckets still need direct validation and benchmarking
- CPU thread-pool tuning is now available but the latest Threadripper HLS runs
  showed that the current aggressive helper-driven profiles can make the live
  path much worse
- earlier poor `8`-stream reality check that motivated the host-side refactor:
  - `avg_time_to_live_ready_s = 3.469`
  - `avg_segment_interval_s = 3.622`
  - `max_segment_interval_s = 6.137`
  - `avg GPU util = 37.2%`
- latest post-refactor `8`-stream reality check:
  - `avg_time_to_live_ready_s = 1.760`
  - `avg_segment_interval_s = 1.733`
  - `max_segment_interval_s = 2.535`
  - `avg GPU util = 82.06%`
- latest March 24 ramp clarification:
  - `concurrency=6`
    - `avg_time_to_live_ready_s = 1.342`
    - `avg_segment_interval_s = 1.294`
    - `max_segment_interval_s = 2.032`
    - `avg GPU util = 83.84%`
  - `concurrency=7`
    - `avg_time_to_live_ready_s = 1.508`
    - `avg_segment_interval_s = 1.516`
    - `max_segment_interval_s = 2.530`
    - `avg GPU util = 84.83%`
  - repeated `concurrency=8`
    - `avg_time_to_live_ready_s = 1.569-1.760`
    - `avg_segment_interval_s = 1.733-1.736`
    - `max_segment_interval_s = 2.524-2.535`
    - `avg GPU util = 82.06-83.64%`
  - later 64-core worker-scale check (`prep/compose/encode = 12/12/12`)
    - `concurrency=8`
      - `avg_time_to_live_ready_s = 1.823`
      - `avg_segment_interval_s = 1.827`
      - `max_segment_interval_s = 2.533`
      - `avg GPU util = 76.83%`
    - `concurrency=10`
      - `avg_time_to_live_ready_s = 1.812`
      - `avg_segment_interval_s = 2.251`
      - `max_segment_interval_s = 3.531`
      - `avg GPU util = 80.33%`
- current interpretation:
  - the first multithreaded host-pipeline refactor slice materially recovered throughput
  - the GPU is no longer obviously starving the way it was in the bad run
  - the path is back in the healthy near-threshold band
  - `concurrency=6` is now the first practical realtime milestone on this branch
    even though the strict load-test warning still trips by about `32ms`
  - `concurrency=7` and `concurrency=8` are now in the same batch-4 saturation
    regime, so they look much closer than you would expect from the raw stream count
  - the later 64-core `12/12/12` worker test did **not** create a new throughput tier
  - keep the `8/8/8` worker profile as the current stable baseline until the
    next encode/compose refactor slices are measured
  - do **not** treat `MUSETALK_CPU_TUNING=1` as the current default for HLS
  - keep CPU tuning disabled for the stable baseline while continuing the host pipeline refactor

Current CPU-tuning helper coverage:

- helper module:
  - `scripts/runtime_cpu_tuning.py`
- wired entrypoints:
  - `api_server.py`
  - `scripts/benchmark_pipeline.py`
  - `scripts/inference.py`
  - `scripts/realtime_inference.py`
  - `scripts/preprocess.py`

Current stable baseline block (recommended after the first host-side refactor slice):

```bash
cd /content/MuseTalk
source /content/py310_trt_exp/bin/activate

unset PYTORCH_CUDA_ALLOC_CONF

export MUSETALK_COMPILE=0
export MUSETALK_COMPILE_UNET=0
export MUSETALK_COMPILE_VAE=0
export MUSETALK_WARM_RUNTIME=1
unset MUSETALK_CPU_TUNING
unset MUSETALK_CPU_THREADS
unset MUSETALK_CPU_INTEROP_THREADS
unset MUSETALK_CPU_CV2_THREADS
unset MUSETALK_CPU_NUMA_NODE
unset MUSETALK_CPU_AFFINITY

export MUSETALK_TRT_ENABLED=1
export MUSETALK_VAE_BACKEND=trt_stagewise
export MUSETALK_TRT_FALLBACK=0
export MUSETALK_TRT_STAGEWISE_TORCH_EXECUTED_OPS=native_group_norm
export MUSETALK_TRT_STAGEWISE_TORCH_STAGES=

export AVATAR_CACHE_MAX_AVATARS=0
export AVATAR_CACHE_MAX_MEMORY_MB=12000
export AVATAR_CACHE_TTL_SECONDS=3600

export HLS_SCHEDULER_MAX_BATCH=4
export HLS_SCHEDULER_FIXED_BATCH_SIZES=4
export HLS_SCHEDULER_STARTUP_SLICE_SIZE=4
export HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS=999
export HLS_STARTUP_CHUNK_DURATION_SECONDS=0.5
export HLS_STARTUP_CHUNK_COUNT=1
export HLS_PREP_WORKERS=8
export HLS_COMPOSE_WORKERS=8
export HLS_ENCODE_WORKERS=8
export HLS_MAX_PENDING_JOBS=24

export HLS_CHUNK_VIDEO_ENCODER=libx264
export HLS_CHUNK_ENCODER_PRESET=ultrafast
unset HLS_CHUNK_ENCODER_TUNE
unset HLS_CHUNK_ENCODER_QP
export HLS_CHUNK_ENCODER_CRF=28
export HLS_PERSISTENT_SEGMENTER=0

export MUSETALK_WHISPER_SEGMENT_BATCH_SIZE=4
export MUSETALK_AVATAR_LOAD_WORKERS=8

python api_server.py --host 0.0.0.0 --port 8000
```

Experimental CPU-tuning block (implemented, but not current default):

```bash
export MUSETALK_CPU_TUNING=1
export MUSETALK_CPU_THREADS=4
export MUSETALK_CPU_INTEROP_THREADS=2
export MUSETALK_CPU_CV2_THREADS=1
# Optional:
# export MUSETALK_CPU_NUMA_NODE=0
# export MUSETALK_CPU_AFFINITY=0-15
```

Look for these lines in the startup logs if you intentionally test the helper:

- `VAE decode backend active: trt_stagewise`
- `torch.compile disabled (set MUSETALK_COMPILE=1 to enable)`
- `CPU tuning enabled for api_server: threads=4, interop=2, cv2=1`
- `CPU runtime tuning active for api_server: ...`
- scheduler settings consistent with:
  - `max_combined_batch_size=4`
  - `fixed_batch_sizes=[4]`

Batch-size note for future `bs8` testing:

- the current live backend is `MUSETALK_VAE_BACKEND=trt_stagewise`
- `trt_stagewise` is **not** the old serialized TRT engine workflow
- the older serialized/exported TRT path from `scripts/tensorrt_export.py` does
  require explicit `--batch-sizes ...` coverage if you want an engine or
  metadata range that includes `8`
- the current `trt_stagewise` backend instead compiles exact batch sizes on
  demand and caches them at runtime in `scripts/trt_runtime.py`
- practical meaning:
  - moving from live `batch_size=4` to live `batch_size=8` does **not** require
    recreating a reusable VAE TRT artifact if you stay on `trt_stagewise`
  - it **does** require correctness validation and ideally a warmup before real
    HLS benchmarking, because stagewise warmup currently precompiles `batch=4`
    first and `batch=8` may otherwise compile on first live use
- current validation command for that gate:

```bash
cd /content/MuseTalk
source /content/py310_trt_exp/bin/activate

python scripts/validate_vae_backend.py \
  --avatar-id test_avatar \
  --backend trt_stagewise \
  --batch-size 8 \
  --output-dir ./tmp/vae_backend_validation_bs8
```

- if the backend is changed back to serialized `trt` / `tensorrt` artifacts
  later, rerun export with `8` included in `--batch-sizes ...` before expecting
  that path to support live `bs8`

If CPU tuning is revisited later, change only one knob at a time:

- `MUSETALK_CPU_THREADS=4 -> 6 -> 8`
- `MUSETALK_CPU_INTEROP_THREADS=2 -> 3 -> 4`
- keep `MUSETALK_CPU_CV2_THREADS=1` unless measurements show compose/prep
  improving without tail-latency regressions

Current host-side refactor order:

1. parallelize shared HLS prep in `scripts/hls_gpu_scheduler.py`
2. reduce avatar cache-miss cost in `scripts/api_avatar.py`
3. pre-encode request audio once and reuse `-c:a copy` during chunk muxing
4. refactor compose to use CPU cores more effectively
5. replace per-chunk `ffmpeg` spawn with a bounded shared encode architecture
6. consolidate older direct streaming paths onto the shared scheduler model

Current code status:

- the first refactor slice is now implemented:
  - batched audio feature extraction / Whisper segment encode
  - vectorized audio prompt construction
  - concurrent HLS prep subtasks
  - parallel avatar cache-miss frame/mask loading
- latest measured `concurrency=8` result after that slice:
  - `completed=8/8`
  - `avg_time_to_live_ready_s=1.760`
  - `avg_segment_interval_s=1.733`
  - `max_segment_interval_s=2.535`
  - `avg GPU util ~= 82.06%`
- latest measured practical realtime milestone:
  - `concurrency=6`
  - `completed=6/6`
  - `avg_time_to_live_ready_s=1.342`
  - `avg_segment_interval_s=1.294`
  - `max_segment_interval_s=2.032`
  - `avg GPU util ~= 83.84%`
- persistent encode is **not** implemented yet in this branch
- reusable AAC sidecar prep is now implemented for the shared HLS path:
  - `scripts/api_avatar.py`
    - prepares a per-request AAC sidecar
    - chunk ffmpeg jobs now try `audio=copy` first and fall back to per-chunk
      AAC encode if needed
  - `scripts/hls_gpu_scheduler.py`
    - prepares the sidecar during HLS prep
    - tracks sidecar prep time in scheduler logs
    - cleans up the sidecar with the request lifecycle
  - this slice has passed targeted smoke validation but has not yet been
    re-benchmarked end-to-end in `load_test.py` during this turn

New optional tuning knobs for the first refactor slice:

- `MUSETALK_WHISPER_SEGMENT_BATCH_SIZE`
  - default: `4`
  - controls how many 30s Whisper mel segments are encoded together
- `MUSETALK_AVATAR_LOAD_WORKERS`
  - default: auto
  - caps parallel avatar frame/mask read workers during cache-miss load
- `HLS_CHUNK_PREPARE_AUDIO_SIDECAR`
  - default: enabled
  - prepares a reusable AAC sidecar once per request so chunk muxing can try
    `-c:a copy`

## What To Verify On Startup

Look for these lines in the server logs:

- `fixed_batch_sizes=[4, 8, 16, 32, 48]`
- `VAE decode backend active: tensorrt`
- `torch.compile disabled (set MUSETALK_COMPILE=1 to enable)`
- `WebRTC disabled (missing deps): No module named 'aiortc'`

## What To Verify During TRT Benchmarking

Look for these lines in the benchmark/runtime logs:

- `TensorRT VAE backend is active`
- `VAE decode backend: tensorrt`

If you instead see `VAE decode backend: pytorch`, the TRT artifact did not
activate in that process.

## Current Backend-Active HLS Result

The first full end-to-end HLS load test with the broad-batch TRT VAE artifact
has now completed in `/content/py310_trt_exp`.

Current saved result:

- file:
  - `load_test_report.json`
- server shape:
  - eager UNet
  - TRT VAE
  - existing prepared avatar: `test_avatar`
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

Important caveat:

- this shows the **full HLS generation path executes** with TRT active
- but it is **not** yet a clean higher-concurrency throughput result because
  ffmpeg repeatedly failed to open `h264_nvenc` and fell back to CPU
  `libx264`
- observed encoder failure included:
  - `OpenEncodeSessionEx failed: unsupported device (2)`
  - `No capable devices found`
  - repeated `Broken pipe` retries before fallback
- and the current active TRT VAE artifact is now known to be visually wrong
  for avatar output, with the talking-face ROI collapsing into a gray mask

Practical meaning:

- TRT VAE is now validated past the isolated benchmark and into real HLS output
- the next meaningful load-test step is to fix or deliberately account for the
  encoder fallback before reading too much into `5`- or `6`-stream results
- separate from that, the current active TRT VAE artifact is now known to be
  visually wrong and should not be treated as a valid avatar-output result

## Current Stagewise HLS Visual Result

A later HLS `/wall` run with the new `trt_stagewise` backend now appears
visually correct on the avatar:

- the mouth region is no longer replaced by the old flat gray ROI
- real lip output is visible again across the wall tiles
- this result lines up with the strong `batch_size=4` decode validation in
  `tmp/vae_stagewise_backend_validation_bs4/report.json`

Current honest caveat:

- this is the first visual success signal, not the final throughput verdict
- stagewise throughput and larger-batch correctness are still the next required
  measurements
- a later real `load_test.py` run on the same stagewise path at
  `concurrency=8`, `batch_size=4`, `playback_fps=24`, `musetalk_fps=12`,
  `hold_seconds=30` now shows materially improved live HLS behavior:
  - `completed=8`
  - `failed=0`
  - `avg_time_to_live_ready_s=1.631`
  - `avg_segment_interval_s=1.769`
  - `max_segment_interval_s=2.524`
  - GPU average util about `83.76%`, peak about `96%`
  - GPU memory stayed around `6742 MB`
- practical meaning of that stagewise `concurrency=8` run:
  - startup/live-ready behavior is far better than the older `~4-5s` band
  - steady-state segment pacing is also better than the earlier stable PyTorch
    `~1.97-2.04s avg / ~3.10-3.22s max` band
  - but the run still technically trips the existing throttling rule because
    `max_segment_interval_s` stayed above the `2.0s` threshold
  - so this is strong progress, not final mission-complete proof yet

## Experimental Toggle

`HLS_PERSISTENT_SEGMENTER` is kept here as a legacy experimental toggle from the earlier persistent-NVENC work.

Historical reason it stayed disabled:

- the persistent segmenter holds NVENC sessions open
- at 8 concurrent streams this can exceed the GPU encoder session limit
- that caused `OpenEncodeSessionEx failed: out of memory (10)` and `Broken pipe` failures in testing

Current code-state caveat:

- the current live code path does **not** appear to read this env var anymore
- so changing it today should be treated as documentation/history, not an active tuning lever

If that persistent segmenter path is ever reintroduced, the old experimental toggle would be:

```bash
export HLS_PERSISTENT_SEGMENTER=1
```

## Recent Operational Findings

- `HLS_SCHEDULER_MAX_BATCH=48` remains the practical ceiling on this GPU for the current HLS path. Testing above `48` did not improve throughput.
- The audio-sidecar path is valid and should log:
  - `Prepared reusable AAC sidecar`
  - `Segment created with ... + aac-copy`
- If those lines appear but throughput still stays around the familiar `~2.0 avg / ~3.1 max` band, AAC re-encode is not the dominant bottleneck.
- The compose-cache / batched-compose refactor is also real, but it did not create a new sustained throughput tier by itself.
- A true in-process PyAV encoder path was investigated and ruled out for this runtime because `h264_nvenc` cannot be used there as a viable replacement for the current `ffmpeg` path.
- The persistent NVENC segmenter experiment was useful diagnostically, but it exhausted encoder sessions at `concurrency=8`, so this file keeps it disabled by default.
- The later GPU-resident conditioning experiment was reverted. It did not improve steady-state throughput and regressed startup fairness toward a `~3s` first stream / `~5s` rest-of-wave pattern, so it is not part of the stable launch config.
- The later vectorized audio-prompt experiment was also rolled back as a throughput change. It preserved behavior correctly, but at `concurrency=8`, `playback_fps=24`, `musetalk_fps=12` it still landed around `avg_segment_interval_s = 2.0`, `max_segment_interval_s = 3.225`, and `avg_time_to_live_ready_s = 4.772`, so it is not part of the stable launch config either.
- The later GPU-resident latent-cycle experiment was also rolled back as a throughput change. Across repeated `concurrency=8`, `playback_fps=24`, `musetalk_fps=12` runs it stayed in the same familiar band at about `avg_segment_interval_s = 1.97-1.99`, `max_segment_interval_s = 3.12-3.14`, and `avg_time_to_live_ready_s = 4.774`, so it is not part of the stable launch config either.
- The later explicit SDPA attention-path experiment was also rolled back as a throughput change. Across repeated `concurrency=8`, `playback_fps=24`, `musetalk_fps=12` runs it stayed in the same familiar band at about `avg_segment_interval_s = 1.97-2.04`, `max_segment_interval_s = 3.10-3.22`, and `avg_time_to_live_ready_s = 4.15-4.77`, so it is not part of the stable launch config either.
- These stable start params were still the correct way to run all of those recent small-model-path experiments. Those tests did not need extra start flags beyond any code-default toggles, so the failed results are still valid evidence.

## Experimental `bs8` Branch Notes

The current default production-like baseline in this file is still the moderate
`bs4` HLS profile. A later March 25 experiment widened the live scheduler to
exercise the repaired `trt_stagewise` backend at `bs8`.

Important config notes from that experiment:

- `HLS_SCHEDULER_STARTUP_SLICE_SIZE` expects a single integer, not a list
- the earlier scratch command `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4,8` was
  malformed and should not be reused
- forcing `HLS_SCHEDULER_FIXED_BATCH_SIZES=8` made even low-concurrency turns
  pad to `8`, which increased resident VRAM sharply
- the safer follow-up experiment is to allow both live buckets:
  - `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8`
  - `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4`

Observed March 25 `bs8` results:

- `concurrency=1`, `batch_size=8`
  - `avg_time_to_live_ready_s=1.006`
  - `avg_segment_interval_s=0.192`
  - `max_segment_interval_s=0.509`
  - `avg GPU util ~= 41.14%`
  - `avg GPU memory used ~= 13742 MB`
- `concurrency=8`, `batch_size=8`
  - `avg_time_to_live_ready_s=1.947`
  - `avg_segment_interval_s=1.513`
  - `max_segment_interval_s=2.531`
  - `wall_time_s=28.4`
  - `avg GPU util ~= 82.87%`
  - `avg GPU memory used ~= 13821 MB`
- later widened `max_batch=16` branch on the same date produced the current
  best average-throughput `concurrency=8` result so far:
  - server-side shape:
    - `HLS_SCHEDULER_MAX_BATCH=16`
    - `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16`
    - `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4`
    - `HLS_PREP_WORKERS=8`
    - `HLS_COMPOSE_WORKERS=8`
    - `HLS_ENCODE_WORKERS=8`
    - `HLS_MAX_PENDING_JOBS=24`
    - `HLS_CHUNK_VIDEO_ENCODER=libx264`
    - `HLS_CHUNK_ENCODER_PRESET=ultrafast`
    - `HLS_CHUNK_ENCODER_CRF=28`
    - `MUSETALK_WHISPER_SEGMENT_BATCH_SIZE=4`
    - `MUSETALK_AVATAR_LOAD_WORKERS=8`
  - measured run:
    - `concurrency=8`
    - request `batch_size=8`
    - `avg_time_to_live_ready_s=2.197`
    - `avg_segment_interval_s=1.408`
    - `max_segment_interval_s=2.525`
    - `wall_time_s=26.4`
    - `avg GPU util ~= 85.21%`
    - `avg GPU memory used ~= 23922 MB`
  - same server branch with request `batch_size=4`:
    - `avg_time_to_live_ready_s=2.199`
    - `avg_segment_interval_s=1.423`
    - `max_segment_interval_s=2.046`
    - `wall_time_s=26.4`
    - `avg GPU util ~= 85.79%`
    - `avg GPU memory used ~= 23922 MB`

Interpretation:

- this was better than the familiar `bs4` `concurrency=8` band on steady-state
  pacing and total wall time
- it did **not** materially improve the stubborn tail, because
  `max_segment_interval_s` stayed around `2.53s`
- it is therefore a promising throughput branch, but not yet a replacement for
  the stable default launch block in this file
- the widened `max_batch=16` branch now improves that story further:
  - request `batch_size=8` is the new best **average-throughput** result on the
    current branch
  - request `batch_size=4` on the same widened server shape still has the
    better tail (`2.046s` vs `2.525s`)
  - practical meaning:
    - `batch_size=8` is currently the best throughput-oriented choice on this
      widened scheduler branch
    - `batch_size=4` remains the more tail-friendly choice on the same branch
  - important caution:
    - this branch is running very close to the 24 GB VRAM ceiling at about
      `23922 MB`, so it should not warm additional large buckets casually

Recommended experimental `bs8` startup block:

```bash
cd /content/MuseTalk
source /content/py310_trt_exp/bin/activate

unset PYTORCH_CUDA_ALLOC_CONF

export MUSETALK_COMPILE=0
export MUSETALK_COMPILE_UNET=0
export MUSETALK_COMPILE_VAE=0
export MUSETALK_WARM_RUNTIME=1

export MUSETALK_TRT_ENABLED=1
export MUSETALK_VAE_BACKEND=trt_stagewise
export MUSETALK_TRT_FALLBACK=0
export MUSETALK_TRT_STAGEWISE_TORCH_EXECUTED_OPS=native_group_norm
export MUSETALK_TRT_STAGEWISE_TORCH_STAGES=

export AVATAR_CACHE_MAX_AVATARS=0
export AVATAR_CACHE_MAX_MEMORY_MB=12000
export AVATAR_CACHE_TTL_SECONDS=3600

export HLS_SCHEDULER_MAX_BATCH=16
export HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16
export HLS_SCHEDULER_STARTUP_SLICE_SIZE=4
export HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS=999
export HLS_STARTUP_CHUNK_DURATION_SECONDS=0.5
export HLS_STARTUP_CHUNK_COUNT=1
export HLS_PREP_WORKERS=8
export HLS_COMPOSE_WORKERS=8
export HLS_ENCODE_WORKERS=8
export HLS_MAX_PENDING_JOBS=24
export MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16

export HLS_CHUNK_VIDEO_ENCODER=libx264
export HLS_CHUNK_ENCODER_PRESET=ultrafast
unset HLS_CHUNK_ENCODER_TUNE
unset HLS_CHUNK_ENCODER_QP
export HLS_CHUNK_ENCODER_CRF=28
export HLS_PERSISTENT_SEGMENTER=0

export MUSETALK_WHISPER_SEGMENT_BATCH_SIZE=4
export MUSETALK_AVATAR_LOAD_WORKERS=8

python api_server.py --host 0.0.0.0 --port 8000
```
