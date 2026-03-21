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
