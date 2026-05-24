# MuseTalk Cross-Server Throughput Findings

## Purpose

This document captures the cross-server throughput investigation for the same
MuseTalk codebase and the same RTX 3090 GPU when running on different CPU
hosts.

The main question from this branch was:

- why does one server perform worse than another even with the same GPU?
- are avatar preparation parameters causing runtime variance?
- are worker-pool settings or code caps preventing higher CPU utilization?

## Bottom Line

The current live HLS path is **not** GPU-only.

It still includes significant CPU-side work:

- audio feature orchestration and prompt construction
- per-frame face composition / blending
- per-chunk `ffmpeg` encoding and muxing
- scheduler queueing and thread-pool orchestration

That means the same RTX 3090 can produce different end-to-end throughput on
different CPUs.

The current cross-server evidence points more toward a **host-side pipeline
difference** than a pure model-speed difference.

## Important Prep-Path Clarifications

Two preparation details from the current codebase are important:

1. In `v15`, `bbox_shift` is ignored during avatar preparation.
2. The `batch_size` passed to `/avatars/prepare` does not materially change how
   the avatar materials are built.

Relevant code path:

- `api_server.py` calls `manager.prepare_avatar(...)`
- `scripts/avatar_manager_parallel.py`
  - `bbox_shift=bbox_shift if self.args.version == "v1" else 0`
- `scripts/api_avatar.py`
  - preparation builds frames, masks, coords, and latents from the input video

Practical meaning:

- if the user prepared a `v15` avatar with a custom `bbox_shift`, that was not
  actually used in the prepared materials
- preparation-time `batch_size` should not be treated as the main explanation
  for the cross-server runtime difference

## Prepared Avatar Facts For `test_avatar`

Current prepared avatar inspected on disk:

- avatar path:
  - `results/v15/avatars/test_avatar`
- stored metadata:
  - `bbox_shift = 0`
  - `version = v15`
- prepared frame count:
  - `600`
- prepared frame resolution:
  - `1024 x 1024`
- prepared mask resolution:
  - `588 x 588`
- latent count:
  - `600`
- latent shape:
  - `(1, 8, 32, 32)` per frame
- average face bbox area:
  - about `117,222` pixels

Practical meaning:

- this avatar is not a small or cheap CPU-side composition target
- each generated face frame is blended back into a full `1024x1024` frame
- each chunk then goes through `ffmpeg` encoding work

## Why The CPU Still Matters

The live path uses the GPU for UNet + VAE, but then still does CPU-heavy work.

Key runtime path:

- `scripts/hls_gpu_scheduler.py`
  - `_run_generation_batch()` runs the GPU forward pass
  - `_dispatch_compose_batch()` pushes frame composition into a CPU pool
  - `_dispatch_encode()` pushes chunk encoding into a worker pool
- `scripts/api_avatar.py`
  - `compose_frame()` blends a talking-face ROI back into the avatar frame
  - `_create_chunk()` launches `ffmpeg` to encode a segment
- `musetalk/utils/blending.py`
  - `get_image_blending()` performs the actual masked image blend

So the live path is:

1. GPU generation
2. CPU frame composition
3. CPU/process-side chunk encoding
4. playlist append / HLS delivery

This is why two servers with the same GPU can still differ.

## Threadripper Server Results

### Baseline Threadripper Result

Hardware:

- CPU: AMD Ryzen Threadripper PRO 3995WX 64-Core
- GPU: RTX 3090

Observed result:

- `avg_time_to_live_ready_s = 5.086`
- `avg_segment_interval_s = 2.156`
- `max_segment_interval_s = 4.03`
- `wall_time_s = 42.5`
- `avg_gpu_util_pct = 72.51`

Interpretation:

- throughput was worse than expected
- tail latency was poor
- GPU was not pinned at full utilization for the whole run
- that suggests the GPU was waiting on host-side work part of the time

### First Worker-Tuning Improvement

The first worker-tuning pass improved the result.

Observed result:

- `avg_time_to_live_ready_s = 5.15`
- `avg_segment_interval_s = 2.097`
- `max_segment_interval_s = 3.537`
- `wall_time_s = 42.0`
- `avg_gpu_util_pct = 72.81`

Interpretation:

- this moved the needle in the right direction
- average segment interval improved
- worst-case segment interval improved materially

### Later More-Aggressive Worker Test

The next more-aggressive worker test regressed startup/tail behavior again.

Observed result:

- `avg_time_to_live_ready_s = 5.966`
- `avg_segment_interval_s = 2.03`
- `max_segment_interval_s = 4.046`
- `wall_time_s = 42.0`
- `avg_gpu_util_pct = 73.06`

Interpretation:

- average interval looked slightly better
- but startup fairness got worse
- tail latency got worse again
- this is consistent with host-side oversubscription / queue contention

## Other Server Observation

The user also reported a second server that was faster with:

- the same RTX 3090 GPU
- a Ryzen 5 5600G CPU

The exact load-test report from that second machine was not captured in this
chat, so this document treats the second-server result as a qualitative
observation rather than a full benchmark record.

Still, that observation is fully plausible given the current pipeline design.

## Isolated Model Benchmark On The Threadripper Server

The most useful follow-up check on March 22, 2026 was to run the isolated
PyTorch model-path benchmark on the slower Threadripper server itself.

Observed result from `scripts/benchmark_pipeline.py`:

- best throughput: `51.1 fps` at `batch_size=32`
- max sustainable fps per stream at `8` concurrent: `6.4`

Per-batch summary:

| Batch size | UNet (ms) | VAE full (ms) | FPS |
|---|---:|---:|---:|
| 4 | 22.35 | 65.04 | 45.8 |
| 8 | 36.31 | 128.00 | 48.7 |
| 16 | 63.74 | 253.34 | 50.5 |
| 24 | 95.91 | 381.29 | 50.3 |
| 32 | 119.92 | 505.75 | 51.1 |
| 40 | 153.15 | 638.95 | 50.5 |
| 48 | 178.51 | 764.93 | 50.9 |

Practical meaning:

- the isolated PyTorch model path on the Threadripper server is effectively in
  the same performance tier as the earlier `~50.9-51.0 fps` benchmark results
- UNet and VAE timings look normal for the current stable stack
- the raw model path is therefore **not** the main explanation for the
  cross-server difference seen in `load_test.py`

This is one of the strongest pieces of evidence in the whole investigation,
because it separates:

1. raw GPU/model throughput
2. end-to-end HLS delivery throughput

The model benchmark staying normal while the live HLS load test is slower means
the gap is much more likely in the host-side pipeline after or around model
inference.

## Likely Reason For Cross-Server Variability

The current explanation with the strongest support is:

- model inference is only part of the live-path cost
- CPU-side blend/encode work still matters a lot
- the 3995WX machine is likely paying more in:
  - memory latency
  - NUMA / CCD crossing
  - thread scheduling overhead
  - host-side subprocess contention

This workload is not a simple “more cores always wins” case.

It mixes:

- Python orchestration
- OpenCV copies / resizes
- masked frame blending
- `ffmpeg` subprocesses
- chunk-ready queueing

Those can perform worse on a very large multi-chip CPU than on a smaller
higher-clocked desktop CPU.

## Worker-Pool Analysis

### What The Code Actually Does

Worker pools are created in `scripts/hls_gpu_scheduler.py`.

Current behavior:

- `prep_executor`
  - uses exactly `max(1, int(prep_workers))`
- `backfill_executor`
  - capped at `max_workers = max(1, min(2, int(prep_workers)))`
- `compose_executor`
  - uses:
    - `max(6, int(compose_workers), min(10, cpu_count // 2))`
- `encode_executor`
  - uses:
    - `max(6, int(encode_workers), min(10, cpu_count // 2))`

Important clarification:

- `compose_workers=12` is **not** capped down to `10`
- because the formula uses `max(...)`, not `min(...)`
- on a large CPU:
  - `compose_workers=12` means `12`
  - `encode_workers=12` means `12`

### Hard / Effective Caps That Do Exist

The code does have some real caps or effective limits:

1. `backfill_executor` is capped at `2`
   - this affects conditioning backfill work
   - it does **not** scale with large `prep_workers`

2. The GPU scheduler itself is single-threaded
   - one scheduler thread decides work and dispatches downstream tasks
   - it cannot create infinite parallel work for compose/encode pools

3. With `concurrency=8`, there are only `8` streams
   - so even if you request `12` compose or encode workers, the amount of
     simultaneously available per-stream work may be below that

4. Compose tasks are generated per selected job per scheduler turn
   - this naturally limits how many compose jobs can be in flight at once

5. Encode tasks only appear when enough frames accumulate for a chunk
   - so encode worker utilization is also limited by chunk readiness

6. `HLS_MAX_PENDING_JOBS`
   - caps the number of queued/active jobs
   - it is **not** a per-pool worker cap

### Practical Conclusion On `12` Workers

There is **no hard code cap** preventing:

- `HLS_PREP_WORKERS=12`
- `HLS_COMPOSE_WORKERS=12`
- `HLS_ENCODE_WORKERS=12`

from being created.

But there **are** practical limits that may prevent all 12 workers from being
fully utilized:

- only `8` streams in the current test
- one scheduler thread producing work
- chunk-based encode dispatch
- compose/encode work supply may simply be too bursty to keep 12 workers busy

So “12 workers configured” does **not** imply “12 workers effectively busy.”

## Why More Workers Can Hurt

The Threadripper experiments suggest that pushing workers too high can regress:

- startup fairness
- tail latency
- worst-case segment interval

Likely reasons:

- more concurrent OpenCV composition work
- more memory-copy pressure
- more `ffmpeg` subprocess overlap
- more NUMA / scheduler overhead

This is why the first tuning pass helped, but the next more-aggressive one got
worse again.

## Later Current-Branch Reality Check

After the later host-pipeline refactor slice repaired the severe CPU-starved
regression, the same Threadripper branch was rechecked with larger worker pools.

Observed result with `prep/compose/encode = 12/12/12`:

- `concurrency=8`
  - `avg_time_to_live_ready_s = 1.823`
  - `avg_segment_interval_s = 1.827`
  - `max_segment_interval_s = 2.533`
  - `avg_gpu_util_pct = 76.83`
- `concurrency=10`
  - `avg_time_to_live_ready_s = 1.812`
  - `avg_segment_interval_s = 2.251`
  - `max_segment_interval_s = 3.531`
  - `avg_gpu_util_pct = 80.33`

Interpretation:

- the repaired branch is far healthier than the earlier CPU-starved state
- but raising worker counts further still did **not** create a new throughput tier
- the current bottleneck is therefore structural tail latency, not just a lack
  of available worker threads
- the stable baseline should remain the moderate `8/8/8` profile until the next
  encode/compose refactor slices are measured

Later widened live `bs8` check on the same branch:

- `concurrency=8`, `batch_size=8`
  - `avg_time_to_live_ready_s = 1.947`
  - `avg_segment_interval_s = 1.513`
  - `max_segment_interval_s = 2.531`
  - `wall_time_s = 28.4`
  - `avg_gpu_util_pct = 82.87`
  - `avg_gpu_memory_used_mb ~= 13821`

Interpretation update:

- widening the live batch regime now looks more promising than scaling worker
  pools further
- the first `bs8` result improved steady-state throughput without solving the
  same old tail
- the next experiment should not force `fixed_batch_sizes=[8]` globally if
  batch `4` is warmed; on the current 24 GB TRT profile, batch `4` is not
  warmed, so the operational bucket set should stay `8,16`

Later widened `max_batch=16` check on the same branch:

- originally measured server-side shape:
  - `HLS_SCHEDULER_MAX_BATCH=16`
  - `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16`
  - `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16`
  - `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4`
  - workers `8/8/8`
- request `batch_size=8`
  - `avg_time_to_live_ready_s = 2.197`
  - `avg_segment_interval_s = 1.408`
  - `max_segment_interval_s = 2.525`
  - `wall_time_s = 26.4`
  - `avg_gpu_util_pct = 85.21`
  - `avg_gpu_memory_used_mb ~= 23922`
- same server branch with request `batch_size=4`
  - `avg_segment_interval_s = 1.423`
  - `max_segment_interval_s = 2.046`

Interpretation update:

- this is now the current best average-throughput result observed on the branch
- the newer total-batch logic matters more than larger worker pools
- request `batch_size=8` is the better throughput choice here
- request `batch_size=4` is still the better tail-latency choice here
- fixed scheduler buckets must stay aligned with warmed TRT buckets; do not use
  an unwarmed `4` bucket in the `throughput_record` profile
- current operational correction: use `HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16`
  with `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16`

Full-rate `30/30` FPS validation on May 10, 2026:

- single stream on the temporary `4,8` profile:
  - `musetalk_fps=30`, `playback_fps=30`, request `batch_size=8`
  - `completed = 1`, `failed = 0`
  - `avg_time_to_live_ready_s = 1.513`
  - `avg_segment_interval_s = 0.476`
  - `max_segment_interval_s = 0.512`
  - `wall_time_s = 9.2`
  - `peak_gpu_memory_used_mb = 13856`
- eight streams on the temporary `4,8` profile:
  - `completed = 8`, `failed = 0`
  - `avg_time_to_live_ready_s = 3.270`
  - `avg_segment_interval_s = 3.826`
  - `max_segment_interval_s = 5.574`
  - `wall_time_s = 69.0`
  - `peak_gpu_memory_used_mb = 13856`
- eight streams on the current `8,16` throughput profile:
  - `completed = 8`, `failed = 0`
  - `avg_time_to_live_ready_s = 5.032`
  - `avg_segment_interval_s = 3.567`
  - `max_segment_interval_s = 6.088`
  - `wall_time_s = 66.6`
  - `peak_gpu_memory_used_mb = 23920`

Interpretation update:

- `30/30` works cleanly for one stream, but it is not viable for the 8-stream
  hosted target on this 24 GB RTX 3090 profile
- moving from `4,8` to `8,16` improved average cadence slightly
  (`3.826s -> 3.567s`) and wall time slightly (`69.0s -> 66.6s`)
- the same move worsened startup (`3.270s -> 5.032s` average live-ready), tail
  (`5.574s -> 6.088s` max interval), and memory (`13856MB -> 23920MB`)
- keep `15/30` as the practical hosted 8-stream target; treat `30/30` as a
  quality mode for low concurrency unless the generation path gets a larger
  throughput improvement

Full-rate `24/24` FPS validation on May 10, 2026:

- three streams on the current `8,16` throughput profile:
  - `musetalk_fps=24`, `playback_fps=24`, request `batch_size=8`
  - `completed = 3`, `failed = 0`
  - `avg_time_to_live_ready_s = 1.848`
  - `avg_segment_interval_s = 1.060`
  - `max_segment_interval_s = 1.527`
  - `wall_time_s = 19.9`
  - `avg_gpu_util_pct = 82.33`
  - `peak_gpu_memory_used_mb = 23922`

Interpretation update:

- `24/24` at `concurrency=3` stayed under the `2.0s` throttle threshold
- this looks like a plausible low-concurrency quality profile on the current
  24 GB `8,16` runtime
- this should not be extrapolated to the 8-stream hosted target without a
  separate 8-stream test

Full-rate `20/20` FPS validation on May 10, 2026:

- five streams on the current `8,16` throughput profile:
  - `musetalk_fps=20`, `playback_fps=20`, request `batch_size=8`
  - `completed = 5`, `failed = 0`
  - `avg_time_to_live_ready_s = 2.014`
  - `avg_segment_interval_s = 1.477`
  - `max_segment_interval_s = 2.550`
  - `wall_time_s = 27.5`
  - `avg_gpu_util_pct = 78.07`
  - `peak_gpu_memory_used_mb = 23922`
- four streams on the current `8,16` throughput profile:
  - `musetalk_fps=20`, `playback_fps=20`, request `batch_size=8`
  - `completed = 4`, `failed = 0`
  - `avg_time_to_live_ready_s = 1.889`
  - `avg_segment_interval_s = 1.188`
  - `max_segment_interval_s = 2.041`
  - `wall_time_s = 22.4`
  - `avg_gpu_util_pct = 77.04`
  - `peak_gpu_memory_used_mb = 23922`

Interpretation update:

- `20/20` at five streams completed but exceeded the `2.0s` tail threshold by
  `0.550s`
- `20/20` at four streams was much closer, exceeding the threshold by only
  `0.041s`
- `20/20` therefore looks like a plausible four-stream quality profile, but
  not a clean no-warning profile under simultaneous burst start

## RTX 4090 WebRTC Throughput Check - May 23, 2026

This pass tested the same MuseTalk server stack on an RTX 4090 to compare
against the saved RTX 3090 WebRTC references.

Hardware/runtime observed:

- GPU: `NVIDIA GeForce RTX 4090`
- GPU memory: `24564 MiB`
- Driver: `565.77`
- Python env: `/workspace/.venvs/musetalk_trt_stagewise`
- Encoder: `WEBRTC_H264_ENCODER=libx264`
- WebRTC request shape: `20/20 fps`, request `batch_size=8`
- Ramp tested: `4,5,6,8`

Avatar prepared for this run:

- avatar id: `gpt_moving_avatar_4090`
- source video: `data/video/chatgpt_moving_vid.mp4`
- prepare endpoint elapsed time: about `2m39s`
- prepared frames: `600`
- avatar artifact size on disk: about `742 MB`
- warmed avatar cache memory: about `3174.8 MB`

### Initial `8,16` Throughput Profile Result

The first RTX 4090 pass used the same operational shape as the 24 GB
throughput profile:

```bash
PROFILE=throughput_record
HLS_SCHEDULER_MAX_BATCH=16
HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16
WEBRTC_H264_ENCODER=libx264
```

Startup facts:

- batch `8` TRT warmup completed in `199.56s`
- batch `16` TRT warmup completed in `253.38s`
- total server startup health wait was about `7m50s`
- idle/runtime memory after startup was already about `21613 MB`

Load-test report:

- `load_test_webrtc_4090_gpt_moving_avatar_20_20_4_5_6_8streams_8_16_libx264_20260523.json`

| Streams | Completed | Failed | Avg live-ready | Avg frame interval | Approx aggregate FPS | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 0 | 18.865s | 0.060s | 66.7 | 24105 MB |
| 5 | 5 | 0 | 3.740s | 0.075s | 66.7 | 24105 MB |
| 6 | 0 | 6 | 0.000s | 0.000s | 0.0 | 24105 MB |
| 8 | 0 | 8 | 0.000s | 0.000s | 0.0 | 23715 MB |

Failure mode:

- stages `4` and `5` completed and showed very strong steady frame cadence
- stage `6` failed before live frames completed
- server logs showed CUDA OOM in the VAE decode path:
  - attempted extra allocation: `512 MB` and then `1024 MB`
  - free memory at failure: about `394.69 MB` to `906.69 MB`
  - process memory in use: about `22.65 GB` to `23.15 GB`
  - stack location: `vae.decode_latents()` through the stagewise TRT decode
    backend

Important interpretation:

- `8,16` are TRT bucket sizes, not a guarantee that VRAM remains under 24 GB
- real VRAM use is the sum of model weights, resident TRT engines/workspaces,
  avatar cache tensors, WebRTC buffers, scheduler queues, compose/encode state,
  and transient VAE decode allocations
- on this RTX 4090 run, warming both `8` and `16` left too little transient
  headroom for concurrent WebRTC generation with the new avatar
- the OOM happened during live decode, not during model load

### Batch-8-Only Isolation Retest

To isolate whether the failure was raw 4090 throughput or `16`-bucket residency,
the server was restarted with only the batch-8 scheduler/TRT path:

```bash
PROFILE=throughput_record
HLS_SCHEDULER_MAX_BATCH=8
HLS_SCHEDULER_FIXED_BATCH_SIZES=8
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8
WEBRTC_H264_ENCODER=libx264
```

Startup and memory facts:

- batch `8` TRT warmup completed in `84.16s`
- total server startup health wait was about `1m41s`
- idle memory before avatar cache was about `9231 MB`
- warmed avatar cache memory was again about `3174.8 MB`
- peak memory during the full WebRTC ramp stayed around `10603 MB`

Load-test report:

- `load_test_webrtc_4090_gpt_moving_avatar_20_20_4_5_6_8streams_8only_libx264_20260523.json`

| Streams | Completed | Failed | Avg live-ready | Avg frame interval | Approx aggregate FPS | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 0 | 4.897s | 0.072s | 55.6 | 10601 MB |
| 5 | 5 | 0 | 4.228s | 0.090s | 55.6 | 10601 MB |
| 6 | 6 | 0 | 5.196s | 0.109s | 55.0 | 10603 MB |
| 8 | 8 | 0 | 7.389s | 0.145s | 55.2 | 10603 MB |

Interpretation:

- the RTX 4090 can complete the full `4,5,6,8` WebRTC ramp with the new
  `gpt_moving_avatar_4090` avatar when the resident batch-16 path is removed
- the `8,16` OOM was therefore primarily a VRAM headroom/runtime residency
  problem, not evidence that 6 or 8 streams are impossible on the 4090
- the `8,16` profile is faster at low concurrency but unsafe at `6+` for this
  WebRTC run because it sits too close to the 24 GB memory wall
- the batch-8-only profile is slower than the `8,16` low-concurrency result but
  materially more stable and leaves far more VRAM headroom

### Batch-8-12 Follow-up Retest

To test an intermediate profile, the server was restarted with resident TRT
buckets `8,12`:

```bash
PROFILE=throughput_record
HLS_SCHEDULER_MAX_BATCH=12
HLS_SCHEDULER_FIXED_BATCH_SIZES=8,12
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,12
WEBRTC_H264_ENCODER=libx264
```

Startup and memory facts:

- batch `8` TRT warmup completed in `83.83s`
- batch `12` TRT warmup completed in `250.54s`
- total server startup health wait was about `5m51s`
- idle memory after startup was about `18527 MB`
- warmed avatar cache memory was again about `3174.8 MB`
- peak memory during the full WebRTC ramp stayed around `20699 MB`

Load-test report:

- `load_test_webrtc_4090_gpt_moving_avatar_20_20_4_5_6_8streams_8_12_libx264_20260523.json`

| Streams | Completed | Failed | Avg live-ready | Avg frame interval | Approx aggregate FPS | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 0 | 4.864s | 0.066s | 60.6 | 20601 MB |
| 5 | 5 | 0 | 3.722s | 0.081s | 61.7 | 20699 MB |
| 6 | 6 | 0 | 4.460s | 0.096s | 62.5 | 20699 MB |
| 8 | 8 | 0 | 7.116s | 0.130s | 61.5 | 20699 MB |

Interpretation:

- the `8,12` profile completed the full `4,5,6,8` WebRTC ramp with no OOM
- throughput improved over batch-8-only by about `9.0%`, `11.0%`, `13.6%`,
  and `11.4%` at `4`, `5`, `6`, and `8` streams respectively
- peak VRAM was about `20.7 GB`, leaving roughly `3.8 GB` of device memory
  headroom on this 24 GB RTX 4090
- this makes `8,12` the best observed compromise so far for this avatar:
  materially faster than batch-8-only, while avoiding the `8,16` live-decode
  OOM seen at `6+` streams

### Batch-8-14 Follow-up Retest

The next intermediate profile tested resident TRT buckets `8,14`:

```bash
PROFILE=throughput_record
HLS_SCHEDULER_MAX_BATCH=14
HLS_SCHEDULER_FIXED_BATCH_SIZES=8,14
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,14
WEBRTC_H264_ENCODER=libx264
```

Startup and memory facts:

- batch `8` TRT warmup completed in `86.77s`
- batch `14` TRT warmup completed in `282.81s`
- total server startup health wait was about `6m25s`
- idle memory after startup was about `20261 MB`
- warmed avatar cache memory was again about `3174.8 MB`
- peak memory during the full WebRTC ramp was about `22563 MB`

Load-test report:

- `load_test_webrtc_4090_gpt_moving_avatar_20_20_4_5_6_8streams_8_14_libx264_20260523.json`

| Streams | Completed | Failed | Avg live-ready | Avg frame interval | Max frame interval | Approx aggregate FPS | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 0 | 4.915s | 0.084s | 30.087s | 47.6 | 22449 MB |
| 5 | 5 | 0 | 3.616s | 0.078s | 0.712s | 64.1 | 22561 MB |
| 6 | 6 | 0 | 4.468s | 0.093s | 0.950s | 64.5 | 22563 MB |
| 8 | 8 | 0 | 6.506s | 0.149s | 30.087s | 53.7 | 22563 MB |

Interpretation:

- the `8,14` profile completed the full `4,5,6,8` WebRTC ramp with no OOM
- it was slightly faster than `8,12` at `5` and `6` streams, by about `3.8%`
  and `3.2%` aggregate throughput respectively
- it was worse than `8,12` at `4` and `8` streams, by about `21.4%` and
  `12.8%` respectively
- it produced large `30.087s` max-frame stalls at `4` and `8` streams
- peak VRAM was about `22.56 GB`, leaving only about `2.0 GB` physical device
  headroom on this 24 GB RTX 4090
- this makes `8,14` usable for this specific run, but not the current preferred
  operating point; `8,12` remains the better balance of throughput, smoothness,
  and memory headroom

### RTX 4090 Results vs RTX 3090 Diagnostic Reference

Closest saved RTX 3090 reference:

- `load_test_webrtc_report_20_20_4_5_6_8streams_8_16_diagnostics_libx264.json`
- profile: `throughput_record`
- TRT warmup/fixed buckets: `8,16`
- encoder: `libx264`
- request shape: `20/20 fps`, request `batch_size=8`

The comparison is not perfectly avatar-identical:

- RTX 4090 run used `gpt_moving_avatar_4090` from `chatgpt_moving_vid.mp4`
- saved RTX 3090 diagnostic reference used the existing `test_avatar`

Still, the request shape, codebase, encoder path, and 24 GB GPU memory class are
close enough to make the throughput direction useful.

Batch-8-only comparison:

| Streams | RTX 4090 batch-8-only | RTX 3090 diagnostic | Aggregate throughput delta |
| ---: | --- | --- | ---: |
| 4 | 4/4, 0.072s avg interval, 55.6 aggregate FPS | 4/4, 0.084s, 47.6 aggregate FPS | +16.7% |
| 5 | 5/5, 0.090s avg interval, 55.6 aggregate FPS | 5/5, 0.110s, 45.5 aggregate FPS | +22.2% |
| 6 | 6/6, 0.109s avg interval, 55.0 aggregate FPS | 6/6, 0.131s, 45.8 aggregate FPS | +20.2% |
| 8 | 8/8, 0.145s avg interval, 55.2 aggregate FPS | 8/8, 0.175s, 45.7 aggregate FPS | +20.7% |

Batch-8-12 comparison:

| Streams | RTX 4090 `8,12` | RTX 3090 diagnostic | Aggregate throughput delta |
| ---: | --- | --- | ---: |
| 4 | 4/4, 0.066s avg interval, 60.6 aggregate FPS | 4/4, 0.084s, 47.6 aggregate FPS | +27.3% |
| 5 | 5/5, 0.081s avg interval, 61.7 aggregate FPS | 5/5, 0.110s, 45.5 aggregate FPS | +35.6% |
| 6 | 6/6, 0.096s avg interval, 62.5 aggregate FPS | 6/6, 0.131s, 45.8 aggregate FPS | +36.5% |
| 8 | 8/8, 0.130s avg interval, 61.5 aggregate FPS | 8/8, 0.175s, 45.7 aggregate FPS | +34.6% |

Batch-8-14 comparison:

| Streams | RTX 4090 `8,14` | RTX 3090 diagnostic | Aggregate throughput delta |
| ---: | --- | --- | ---: |
| 4 | 4/4, 0.084s avg interval, 47.6 aggregate FPS | 4/4, 0.084s, 47.6 aggregate FPS | +0.0% |
| 5 | 5/5, 0.078s avg interval, 64.1 aggregate FPS | 5/5, 0.110s, 45.5 aggregate FPS | +41.0% |
| 6 | 6/6, 0.093s avg interval, 64.5 aggregate FPS | 6/6, 0.131s, 45.8 aggregate FPS | +40.9% |
| 8 | 8/8, 0.149s avg interval, 53.7 aggregate FPS | 8/8, 0.175s, 45.7 aggregate FPS | +17.4% |

Operational call from this test:

- for RTX 4090 WebRTC with this avatar, prefer the `8,12` profile when using
  this tested request shape and when about `20.7 GB` peak VRAM is acceptable
- treat `8,14` as experimental/edge: it completed, but had less headroom and
  worse smoothness at the `4` and `8` stream points
- keep batch-8-only as the conservative fallback when lower startup latency,
  lower steady VRAM, or maximum memory headroom matters more than throughput
- do not treat the older 24 GB `8,16` guidance as universally safe across HLS
  and WebRTC paths
- `8,16` can improve steady cadence when it fits, but it can also push a 24 GB
  card so close to the memory wall that a transient VAE decode allocation fails
- scheduler/admission should separate "bucket available" from "enough transient
  headroom remains for this concurrent live decode"

## RTX 6000 Ada WebRTC Throughput Check - May 24, 2026

This pass tested the same MuseTalk WebRTC path on an RTX 6000 Ada 48 GB node to
compare against the saved RTX 3090 and RTX 4090 WebRTC references.

Environment:

- GPU: `NVIDIA RTX 6000 Ada Generation`
- visible VRAM: `49140 MB`
- server profile family: `throughput_record`
- WebRTC encoder: `libx264`
- playback fps / MuseTalk fps: `20/20`
- request batch size: `8`
- avatar: `test_avatar`
- load-test ramp: `4,8,12,16`

The auto-selected 48 GB profile would include `4,8,16,32,48`, but the explicit
test focused on smaller resident TRT buckets first. The known-good `8,16`
profile started successfully:

- batch `8` warmup: `131.74s`
- batch `16` warmup: `266.68s`
- total stagewise warmup: `398.42s`
- health after start: `6m54s`
- idle VRAM after health: roughly `21.7 GB`
- scheduler: `max_combined_batch_size=16`, `fixed_batch_sizes=[8, 16]`

`8,16,32` was then tested and failed during startup. The failure was not printed
as literal PyTorch `CUDA out of memory`, but it is operationally equivalent for
this profile: TensorRT could not create the execution context after VRAM climbed
near the memory wall.

Observed failure:

```text
RuntimeError: [Error thrown at core/runtime/TRTEngine.cpp:93] Expected (exec_ctx.get() != nullptr) to be true but got false
Unable to create TensorRT execution context
ERROR:    Application startup failed. Exiting.
```

A fallback `8,16,24` profile did start successfully:

- batch `8` warmup: `88.94s`
- batch `16` warmup: `101.43s`
- batch `24` warmup: `360.02s`
- total stagewise warmup: `550.40s`
- health after start: `9m25s`
- idle VRAM after health: roughly `40.1 GB`
- scheduler: `max_combined_batch_size=24`, `fixed_batch_sizes=[8, 16, 24]`

### RTX 6000 Ada `8,16` Results

Report:
`load_test_webrtc_rtx6000ada_20_20_4_8_12_16streams_8_16_libx264_20260524.json`

| Streams | Completed | Failed | Avg live-ready | Avg frame interval | Max frame interval | Wall time | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 0 | `16.703s` | `0.051s` | `0.081s` | `35.7s` | `24432 MB` |
| 8 | 8 | 0 | `4.511s` | `0.105s` | `1.336s` | `43.4s` | `24652 MB` |
| 12 | 12 | 0 | `6.318s` | `0.161s` | `2.730s` | `65.2s` | `24602 MB` |
| 16 | 16 | 0 | `9.636s` | `0.219s` | `3.924s` | `89.7s` | `24604 MB` |

### RTX 6000 Ada `8,16,24` Results

Report:
`load_test_webrtc_rtx6000ada_20_20_4_8_12_16streams_8_16_24_libx264_20260524.json`

| Streams | Completed | Failed | Avg live-ready | Avg frame interval | Max frame interval | Wall time | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 0 | `4.176s` | `0.052s` | `0.448s` | `24.2s` | `44270 MB` |
| 8 | 8 | 0 | `4.601s` | `0.103s` | `1.723s` | `42.8s` | `44424 MB` |
| 12 | 12 | 0 | `7.066s` | `0.158s` | `3.588s` | `65.9s` | `44434 MB` |
| 16 | 16 | 0 | `8.389s` | `0.218s` | `5.186s` | `89.3s` | `44422 MB` |

Operational read:

- RTX 6000 Ada is materially faster than the saved RTX 3090 and RTX 4090
  references at 8 streams for this WebRTC path.
- The `8,16` profile reached about `76.2` aggregate FPS at 8 streams
  (`8 / 0.105s`), compared with saved references of about `45.7` aggregate FPS
  on RTX 3090, `55.2` on RTX 4090 batch-8-only, and `61.5` on RTX 4090 `8,12`.
- `8,16,24` is not a clear serving win over `8,16`: average cadence was nearly
  unchanged, tail jitter was worse, and resident VRAM rose from about `24.6 GB`
  to about `44.4 GB`.
- Recommended serving profile from this pass is still `8,16`; treat
  `8,16,24` as a stress-test profile and `8,16,32` as not viable under the
  current runtime shape.

### RTX 6000 Ada `8,16,24` Higher-Concurrency Ramp

After the profile comparison above, the live server was left running on
`8,16,24` and tested with a higher WebRTC concurrency ramp. This was a stress
test of completion capacity, not the recommended production serving profile.

Report:
`load_test_webrtc_rtx6000ada_20_20_10_15_20streams_8_16_24_libx264_20260524.json`

Detailed report:
`load_test_webrtc_rtx6000ada_20_20_10_15_20streams_8_16_24_libx264_20260524_detailed.json`

| Streams | Completed | Failed | Avg live-ready | Avg frame interval | Max frame interval | Wall time | Peak VRAM | Approx aggregate FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 10 | 0 | `5.357s` | `0.123s` | `2.590s` | `51.0s` | `44422 MB` | `81.3` |
| 15 | 15 | 0 | `8.065s` | `0.195s` | `4.610s` | `80.2s` | `44422 MB` | `76.9` |
| 20 | 20 | 0 | `10.715s` | `0.272s` | `6.541s` | `109.8s` | `44422 MB` | `73.5` |

Operational read from the higher-concurrency ramp:

- The RTX 6000 Ada completed `10`, `15`, and `20` concurrent WebRTC streams with
  zero failed sessions on the live `8,16,24` profile.
- Completion capacity is higher than the earlier 3090/4090 references, but
  smooth realtime playback is not sustained at these higher counts.
- Average aggregate throughput peaked around the 10-stream point in this ramp
  (`~81 aggregate FPS`) and then declined as concurrency increased.
- The 15- and 20-stream stages are useful as overload/completion tests, but the
  live frame intervals (`0.195s` and `0.272s`) and tail intervals (`4.610s` and
  `6.541s`) are too high for a smooth 20 fps user-facing target.
- VRAM stayed pinned around `44.4 GB`, so this profile still has very little
  headroom for transient allocations or additional resident buckets.

## Current Practical Guidance

For this cross-server branch:

- treat the workload as mixed GPU + CPU, not GPU-only
- do not assume the Threadripper should win automatically
- do not assume more workers always helps
- treat moderate worker tuning as a queue-management tool, not a model-throughput solution
- on 24 GB WebRTC profiles, treat resident TRT bucket choices as a VRAM
  admission decision, not just a throughput decision; the May 23 RTX 4090 run
  shows `8,16` can OOM while `8,14`, `8,12`, and batch `8` alone complete the
  same `4,5,6,8` ramp

Most likely safe tuning range:

- `HLS_PREP_WORKERS`: modest increase can help
- `HLS_COMPOSE_WORKERS`: small increases can help, large increases can regress
- `HLS_ENCODE_WORKERS`: small increases can help, large increases can regress

## Next Best Diagnostic

The most useful next comparison is to inspect the server-side scheduler finish
logs for:

- `avg_compose`
- `avg_compose_wait`
- `avg_encode`
- `avg_encode_wait`

If those are high on the slower server, that strongly confirms the host-side
pipeline as the cause of the variance.

## Final Summary

Current best explanation:

- the current MuseTalk HLS path still depends materially on CPU-side compose and
  encode work
- the isolated PyTorch model benchmark on the Threadripper server still lands at
  about `51.1 fps`, which means the raw model path is healthy there
- the prepared avatar is large enough to make that CPU work expensive
- the Threadripper server likely loses on host-side pipeline behavior, not on
  raw GPU inference
- there is no hard worker cap preventing `12` compose/encode workers from being
  created, but there are strong practical limits on whether they can actually be
  kept busy
- this is why moderate worker tuning helped briefly, but larger increases
  regressed again
