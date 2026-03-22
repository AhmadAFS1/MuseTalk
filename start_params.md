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

## Latest Benchmark Result

Latest isolated model-path result from `scripts/benchmark_pipeline.py`:

- best throughput: `51.0 fps` at `batch_size=16`
- max sustainable fps per stream at `8` concurrent: `6.4 fps`

Most important interpretation:

- this is below the `96 fps` needed for `8 x 12 fps`
- so the current PyTorch model path is a hard ceiling before HLS compose/encode overhead is even added
- VAE decode is the dominant model-side bottleneck

That is why the next serious branch is now backend acceleration, with VAE first.

## Current Backend Branch State

Repo status today:

- VAE TensorRT runtime wiring is in the codebase
- VAE TensorRT export script is in the codebase
- local TensorRT packages are now installed:
  - `torch-tensorrt==1.4.0`
  - `tensorrt_bindings==8.6.1`
  - `tensorrt_libs==8.6.1`
- repo-root compatibility shim `tensorrt.py` is now in the codebase so local repo scripts can import TensorRT in this environment
- VAE engine has now been exported:
  - `models/tensorrt/vae_decoder_trt.ts`
  - `models/tensorrt/vae_decoder_trt_meta.json`
  - compile time was about `807.4s`
  - engine size is about `141 MB`
- backend-active benchmark attempt happened, but it fell back to PyTorch
- no backend-active `load_test.py` run has happened yet

So the branch crossed the first milestone by producing an engine, but the current environment still did **not** deliver a real TensorRT throughput result. The attempted benchmark fell back to PyTorch, and a stricter re-export then showed this environment is blocked on unsupported operators.

## Current TensorRT Export State

Current VAE export history:

- earlier export attempts failed through the old full-graph path
- after the exporter was patched to use the narrower decoder-only TorchScript path, the next export succeeded
- successful result:
  - `models/tensorrt/vae_decoder_trt.ts`
  - `models/tensorrt/vae_decoder_trt_meta.json`
  - compile time: about `807.4s`
  - saved size: about `141 MB`

Practical meaning:

- install is no longer the blocker
- export is no longer the only blocker
- the current environment is now the blocker
- the attempted benchmark still landed at about `50.9 fps`, which is effectively the same as the old PyTorch baseline
- there is still no trustworthy backend-active throughput result yet

One extra log clarification:

- the TensorFlow `TF-TRT Warning: Could not find TensorRT` messages are noisy and are **not** the current blocker for the MuseTalk scripts

Current recommendation:

- do **not** switch the stable server start script to TensorRT on this environment
- keep the stable start config on the normal PyTorch VAE path for now
- move the next TensorRT attempt into a separate dedicated environment

If a future separate TensorRT environment becomes ready to test, the server config there will need to change from the stable baseline. In particular:

- `MUSETALK_TRT_ENABLED=1`
- `MUSETALK_VAE_BACKEND=trt`
- `MUSETALK_COMPILE_VAE=0`

## Current Environment Failure Record

Benchmark attempt with TensorRT requested:

- backend activation failed with:
  - `Unknown type name '__torch__.torch.classes.tensorrt.Engine'`
- benchmark log showed:
  - `VAE decode backend: pytorch`
- resulting throughput stayed in the same familiar band:
  - best throughput: `50.9 fps`
  - max sustainable fps per stream at `8` concurrent: `6.4 fps`

After loader fixes, strict full re-export failed on unsupported operators:

- `aten::scaled_dot_product_attention`
- `aten::group_norm`

So the current environment should now be treated as blocked for this VAE TensorRT branch.

## Previous Test Commands

Benchmark the model path with the TensorRT VAE backend active:

```bash
cd /content/MuseTalk
MUSETALK_TRT_ENABLED=1 \
MUSETALK_VAE_BACKEND=trt \
MUSETALK_COMPILE_VAE=0 \
/content/py310/bin/python scripts/benchmark_pipeline.py \
  --batch-sizes 4,8,16,24,32,40,48 \
  --warmup 20 \
  --iters 50 \
  --output-json benchmark_pipeline_trt_vae.json
```

If a future separate TensorRT environment improves materially, then start the server there with:

```bash
export MUSETALK_TRT_ENABLED=1
export MUSETALK_VAE_BACKEND=trt
export MUSETALK_COMPILE_VAE=0
```

and rerun the normal HLS `load_test.py`.

## What To Verify On Startup

Look for these lines in the server logs:

- `fixed_batch_sizes=[4, 8, 16, 32, 48]`
- `encode_workers=8`
- `UNet warmup bs=4, 8, 16, 32, 48`
- `VAE warmup bs=4, 8, 16, 32, 48`

## What To Verify During The Refactor Test

For the current audio-sidecar / chunk-encode refactor, look for:

- `Prepared reusable AAC sidecar`
- `Segment created with ... + aac-copy`

If those appear, the new chunk audio-copy path is active.

## Experimental Toggle

`HLS_PERSISTENT_SEGMENTER` is currently disabled by default for the 8-stream NVENC target.

Why:

- the persistent segmenter holds NVENC sessions open
- at 8 concurrent streams this can exceed the GPU encoder session limit
- that caused `OpenEncodeSessionEx failed: out of memory (10)` and `Broken pipe` failures in testing

If you explicitly want to experiment with it again, set:

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
