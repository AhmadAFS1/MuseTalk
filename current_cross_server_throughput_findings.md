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
- the next experiment should not force `fixed_batch_sizes=[8]` globally;
  it should use mixed `4,8` buckets

Later widened `max_batch=16` check on the same branch:

- server-side shape:
  - `HLS_SCHEDULER_MAX_BATCH=16`
  - `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16`
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

## Current Practical Guidance

For this cross-server branch:

- treat the workload as mixed GPU + CPU, not GPU-only
- do not assume the Threadripper should win automatically
- do not assume more workers always helps
- treat moderate worker tuning as a queue-management tool, not a model-throughput solution

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
