# MuseTalk Concurrency Hardening And Scaling Plan

## Purpose

This document describes:

1. What is currently going wrong under load.
2. Why the current backend stalls and keeps GPU utilization high after sessions are deleted.
3. The most robust architecture to fix it.
4. A phased implementation plan, from immediate hardening to a production-grade design.

The goal is not just to "stop leaks". The goal is to make HLS and SSE streaming predictable under load so the backend can support the maximum safe number of simultaneous users on a single GPU, and later scale across multiple GPUs or hosts. WebRTC exists in the codebase, but it is not the current optimization focus.

## Executive Summary

The current bottleneck is primarily a lifecycle problem, not a Python garbage collection problem.

The most important issue is:

- HLS session deletion removes session metadata and files, but does not stop the running inference worker.

That creates three concrete problems:

1. Executor worker threads keep running after the client has timed out or deleted the session.
2. GPU allocation budget stays held until those workers finish naturally.
3. New requests arrive while stale workers still own GPU time and worker slots, so the next run appears "stuck" in audio processing or chunk generation.

There are also secondary issues:

1. Same-avatar cache misses are not deduplicated, so multiple concurrent requests can load the same avatar from disk at the same time.
2. The streaming path uses a shared per-avatar temp directory, which is unsafe under concurrent requests for the same avatar.
3. The HLS, SSE, and WebRTC paths all share the same executor and shared model objects, so one bad workload can affect the others.
4. Request tracking is incomplete. Active requests are inserted into `manager.active_requests`, but HLS and WebRTC flows are not fully reaped.
5. The current GPU memory manager is a logical counter, not a true cancellation or recovery mechanism.

## Investigation Timeline And What We Learned

This section captures the actual load-test findings from the debugging cycle that led to this document.

### Initial symptom set

Observed behavior under HLS load:

1. Sessions timed out after several minutes.
2. Session delete logs appeared, but `active_requests` stayed non-zero.
3. GPU utilization remained high after the client considered the run finished.
4. A second load test often started in a contaminated state and looked "stuck" much earlier.

Initial conclusion:

- the backend was not fully stopping active HLS work when sessions were deleted or when the client timed out.

### Confirmed lifecycle problem

The strongest evidence was:

1. After failed runs, new requests started with a high carried-over logical GPU allocation instead of returning to baseline.
2. Deleting HLS sessions removed session metadata and files, but the executor worker kept running.
3. `manager.active_requests` still reported active jobs after the sessions were deleted.

Conclusion:

- there was a real orphaned-worker problem.

### First round of fixes already implemented

The following hardening changes were implemented in the codebase during this investigation:

1. HLS delete now requests cancellation before final deletion.
2. Active HLS requests now receive a `cancel_event`.
3. `APIAvatar.inference_streaming()` now checks for cancellation during setup, audio processing, generation, and chunk creation.
4. Finished HLS requests now remove themselves from `manager.active_requests`.
5. HLS session deletion can defer final removal until the worker actually exits.
6. Same-avatar cold loads are deduplicated with per-avatar load locks.
7. HLS cleanup logic no longer deletes active sessions during TTL cleanup.
8. Generation progress logging was added so "slow" can be distinguished from "silent".

These changes were necessary. They removed the worst session-lifecycle contamination and made cancellation behavior materially safer.

### What changed after the cancellation fix

After the first lifecycle fix, the failure mode changed:

1. Fresh high-concurrency runs no longer obviously started with stale GPU allocation from the previous run.
2. Requests could now progress further into live generation before failing or timing out.
3. At `concurrency=4`, all requests were able to reach phase 2 generation, but they still stalled before useful HLS output.

Conclusion:

- the system had both a lifecycle problem and a throughput problem.
- fixing delete/cancel behavior was necessary, but not sufficient.

### Second round of changes already implemented

The next structural change was to remove the old HLS model of "one GPU-driving worker thread per stream".

The following changes are now implemented:

1. HLS streaming now uses a shared scheduler in `scripts/hls_gpu_scheduler.py`.
2. Audio preparation for HLS jobs runs in a dedicated prep pool instead of the main request thread.
3. GPU generation for HLS jobs is driven by a single scheduler thread that batches work across active HLS streams.
4. HLS chunk encoding now runs in a separate encode worker pool instead of blocking the GPU-driving loop.
5. HLS requests now move through explicit states such as `preparing`, `queued`, `generating`, and `streaming`.
6. HLS queue pressure is now exposed through scheduler stats and can reject new streams only when the scheduler queue is full.
7. Streaming scratch space is now request-scoped instead of shared per avatar.

Conclusion:

- HLS no longer oversubscribes the GPU by launching one independent live generation loop per stream.
- CPU encode work is now separated from GPU generation work.
- The backend now has a foundation for true multi-session HLS scheduling rather than blind thread-level concurrency.

### What `concurrency=4` proved

Observed behavior after the cleanup fixes:

1. Requests reached phase 2.
2. Generation loop started for all four requests.
3. Progress logs showed the first batch, but the streams still did not become healthy live streams.

Conclusion:

- `concurrency=4` is beyond the practical live HLS throughput limit of the current architecture on one GPU.
- this is not mainly a VRAM-capacity issue.
- this is a scheduling and throughput issue.

### What `concurrency=2` proved

Observed behavior:

1. Both requests reached batch progress.
2. Both requests created chunk 1 successfully.
3. GPU utilization reached 100 percent during active generation.

Conclusion:

- the backend was not frozen in the strict sense.
- 100 percent GPU utilization during this phase is expected.
- if the client still looked frozen at first frame, the issue was likely live-start buffering or slow time-to-next-chunk, not immediate worker death.
- after the scheduler change, `concurrency=2` should be evaluated again against shared-batch throughput, not against the old per-request-thread design.

### What `concurrency=1` proved

This was the most important load-test result.

Observed metrics for one live HLS stream:

1. `live_ready` was about 37.9 seconds.
2. Average segment interval was about 3.18 seconds.
3. Maximum segment interval was about 3.63 seconds.
4. Total wall time was about 91.4 seconds for about 17.7 seconds of generated content.

Conclusion:

- the single-stream HLS pipeline is already slower than real time.
- the main bottleneck is not "how many requests fit in VRAM".
- the main bottleneck is end-to-end throughput per live stream.

That means concurrency tuning alone cannot solve the problem. Even one stream needs optimization or a different operational target.

### March 12 scheduler-era measurements

After the shared HLS scheduler was introduced, the backend was retested with:

- `segment_duration=2.0`
- `musetalk_fps=12`
- `batch_size=4`

These results are materially better than the older baseline, but still not at a healthy realtime target.

#### `concurrency=1`

Observed metrics:

1. `live_ready` was about 5.17 seconds.
2. Average segment interval was about 4.09 seconds.
3. Maximum segment interval was about 4.22 seconds.
4. Total wall time was about 37.0 seconds.

Interpretation:

- this is a major improvement over the earlier single-stream baseline
- however, with `segment_duration=2.0`, the target cadence is still roughly one segment every 2 seconds
- the observed cadence is still about 2x slower than the realtime target

Conclusion:

- the new scheduler improves first-chunk readiness and overall completion time
- but even the improved single-stream path remains throughput-limited

#### `concurrency=2`

Observed metrics:

1. `live_ready` averaged about 11.07 seconds.
2. Average segment interval was about 7.52 seconds.
3. Maximum segment interval was about 8.01 seconds.
4. Total wall time was about 72.9 seconds.

Interpretation:

- both streams completed cleanly
- scheduler fairness is good enough that both sessions make progress
- but throughput collapses under shared load and is still far from realtime

Conclusion:

- `concurrency=2` is functionally supported in the sense that both jobs complete
- `concurrency=2` is not yet operationally healthy for realtime HLS

#### `concurrency=3`

Observed metrics:

1. `live_ready` averaged about 29.41 seconds.
2. Two sessions became live in about 8 to 9 seconds.
3. One session starved and did not become live until about 71.15 seconds.
4. Average segment interval was about 6.26 seconds.
5. Maximum segment interval was about 7.49 seconds.
6. Total wall time was about 103.8 seconds.

Interpretation:

- the scheduler can eventually complete three jobs
- fairness is not strong enough under three-way contention
- tail latency becomes unacceptable
- one late job can wait a very long time before receiving enough shared GPU service to become live

Conclusion:

- `concurrency=3` is beyond the practical realtime limit of the current HLS scheduler settings
- the next optimization target is not just raw speed, but also queueing policy and fairness under contention

### March 12 post chunk-boundary and encode-path measurements

After the later March 12 scheduler fix, the backend was retested again.

The two relevant changes were:

1. the shared HLS scheduler now flushes exact chunk boundaries instead of occasionally handing an oversized frame buffer to one encode task
2. HLS chunk encoding now prefers NVENC with fallback to `libx264`, which reduces CPU-side encode pressure where NVENC is available

These runs used the current `load_test.py` defaults plus `batch_size=4`:

- `segment_duration=1.0`
- `playback_fps=30`
- `musetalk_fps=15`
- `batch_size=4`

#### `concurrency=1`

Observed metrics:

1. `live_ready` was about 1.53 seconds.
2. Average segment interval was about 0.81 seconds.
3. Maximum segment interval was about 1.08 seconds.
4. Total wall time was about 15.0 seconds.

Interpretation:

- single-stream HLS is now comfortably inside the 1-second segment target envelope
- time to first visible live output is materially improved
- the system no longer shows single-stream throttling under this test profile

Conclusion:

- `concurrency=1` is now operationally healthy for realtime HLS under the tested settings

#### `concurrency=2`

Observed metrics:

1. `live_ready` averaged about 2.35 seconds.
2. Average segment interval was about 1.62 seconds.
3. Maximum segment interval was about 2.16 seconds.
4. Total wall time was about 29.2 seconds.

Interpretation:

- both streams complete cleanly and become live quickly
- average cadence is much better than the earlier scheduler-era baseline
- the run is near the current throttling threshold, because max interval slightly exceeds the 2.0-second alert threshold for 1-second segments

Conclusion:

- `concurrency=2` is now practically usable
- `concurrency=2` is not yet comfortably inside the realtime safety margin

#### `concurrency=3`

Observed metrics:

1. `live_ready` averaged about 2.89 seconds.
2. Average segment interval was about 2.43 seconds.
3. Maximum segment interval was about 2.74 seconds.
4. Total wall time was about 44.4 seconds.
5. All three sessions completed without failures.

Interpretation:

- fairness is materially better than the earlier starved-session behavior
- the scheduler can keep all three sessions moving
- however, chunk cadence is now clearly beyond the realtime target

Conclusion:

- `concurrency=3` is functionally supported in the sense that all jobs complete
- `concurrency=3` is still throttled for realtime HLS
- the next optimization target is three-way shared-load cadence, not basic correctness

#### `concurrency=4`

Observed metrics:

1. `live_ready` averaged about 3.36 seconds.
2. Average segment interval was about 3.26 seconds.
3. Maximum segment interval was about 3.72 seconds.
4. Total wall time was about 59.7 seconds.
5. All four sessions completed without failures.

Interpretation:

- startup fairness remains acceptable, since all four sessions became live in about 2.6 to 4.2 seconds
- steady-state cadence is now far beyond the 1-second target
- this continues to look less like a startup problem and more like a fixed aggregate throughput ceiling being shared across more sessions

Conclusion:

- `concurrency=4` is functionally reliable, but not operationally healthy for realtime HLS
- `concurrency=4` is useful as a scheduler-fairness and shared-batch stress case, not a production target under current settings

#### `concurrency=5`

Observed metrics:

1. `live_ready` averaged about 4.12 seconds.
2. Average segment interval was about 4.11 seconds.
3. Maximum segment interval was about 4.71 seconds.
4. Total wall time was about 76.6 seconds.
5. All five sessions completed without failures.

Interpretation:

- the scheduler can still bring all five sessions live without failures
- startup latency is still manageable, but steady-state delivery is now roughly 4x slower than the 1-second segment target
- additional concurrency is mainly time-slicing the same saturated throughput budget instead of materially increasing total realtime capacity

Conclusion:

- `concurrency=5` is beyond the current practical live capacity of one GPU under these settings
- the remaining challenge is now raw aggregate throughput, not lifecycle robustness

#### `concurrency=6`

Observed metrics:

1. `live_ready` averaged about 4.85 seconds.
2. Average segment interval was about 4.94 seconds.
3. Maximum segment interval was about 5.39 seconds.
4. Total wall time was about 90.8 seconds.
5. All six sessions completed without failures.

Interpretation:

- the system remains stable enough to complete all sessions, which is a major lifecycle win compared with the older failure modes
- however, the segment cadence shows the backend is now deeply throughput-limited
- each additional concurrent stream adds little realtime capacity and mostly stretches completion time

Conclusion:

- `concurrency=6` confirms the scheduler is robust but the single-GPU throughput ceiling has been exceeded by a wide margin
- meaningful gains beyond `concurrency=2` or `3` will require better aggregate GPU throughput, not just higher admission counts

#### High-concurrency follow-up with `batch_size=2`

Follow-up run settings:

1. `LIVE_MAX_CONCURRENT_GENERATIONS=9`
2. `HLS_SCHEDULER_MAX_BATCH=20`
3. `musetalk_fps=15`
4. `segment_duration=1.0`

#### `concurrency=4` with `batch_size=2`

Observed metrics:

1. `live_ready` averaged about 4.67 seconds.
2. Average segment interval was about 3.21 seconds.
3. Maximum segment interval was about 3.79 seconds.
4. Total wall time was about 59.2 seconds.
5. All four sessions completed without failures.

Comparison against the earlier `batch_size=4` run at the same concurrency:

1. `avg_segment_interval_s` stayed essentially flat (`3.21s` vs `3.26s`).
2. `wall_time_s` stayed essentially flat (`59.2s` vs `59.7s`).
3. `live_ready` was worse at `batch_size=2` (`4.67s` vs `3.36s`).

Interpretation:

- smaller per-stream batches do not improve aggregate throughput at `concurrency=4`
- they may slightly hurt startup time under the current scheduler behavior
- steady-state delivery remains governed by the same shared throughput ceiling

#### `concurrency=6` with `batch_size=2`

Observed metrics:

1. `live_ready` averaged about 5.50 seconds.
2. Average segment interval was about 4.90 seconds.
3. Maximum segment interval was about 5.72 seconds.
4. Total wall time was about 90.3 seconds.
5. All six sessions completed without failures.

Comparison against the earlier `batch_size=4` run at the same concurrency:

1. `avg_segment_interval_s` stayed essentially flat (`4.90s` vs `4.94s`).
2. `wall_time_s` stayed essentially flat (`90.3s` vs `90.8s`).
3. `live_ready` was slightly worse at `batch_size=2`, but not dramatically so.

Interpretation:

- this strongly suggests per-stream `batch_size` is no longer the dominant limiter in the shared HLS path
- aggregate throughput appears capped elsewhere, most likely in the shared GPU/model path rather than in the per-stream slice size
- smaller per-stream batches may still be useful because they reduce per-job footprint and can improve flexibility, even if they do not increase realtime capacity by themselves

#### `concurrency=7` with `batch_size=2`

Observed metrics:

1. `live_ready` averaged about 19.07 seconds.
2. Average segment interval was about 5.74 seconds.
3. Maximum segment interval was about 7.04 seconds.
4. Total wall time was about 120.0 seconds.
5. All seven sessions completed without failures.

Interpretation:

- this is the first run where startup latency degrades catastrophically rather than merely linearly
- the scheduler still completes all sessions, but first-live fairness is now unstable under overload
- by this point the system is operating far outside the realtime envelope

#### `concurrency=7` with `batch_size=2` after scheduler tuning

Follow-up run settings:

1. `LIVE_MAX_CONCURRENT_GENERATIONS=9`
2. `HLS_SCHEDULER_MAX_BATCH=20`
3. startup-fairness scheduling enabled
4. limited aggressive fill for warmed streams enabled

Observed metrics:

1. `live_ready` averaged about 6.04 seconds.
2. Average segment interval was about 5.80 seconds.
3. Maximum segment interval was about 6.91 seconds.
4. Total wall time was about 106.7 seconds.
5. All seven sessions completed without failures.

Comparison against the earlier `concurrency=7` `batch_size=2` run:

1. `avg_time_to_live_ready_s` improved dramatically (`19.07s` -> `6.04s`).
2. `wall_time_s` improved materially (`120.0s` -> `106.7s`).
3. `max_segment_interval_s` improved slightly (`7.04s` -> `6.91s`).
4. `avg_segment_interval_s` became slightly worse (`5.74s` -> `5.80s`), so steady-state throughput did not materially improve.

Interpretation:

- the scheduler tuning clearly improved startup fairness and reduced total completion time
- the startup speedup came from giving not-yet-live jobs a small fairness slice before warmed streams can consume the rest of a scheduler turn
- the newer warmed-stream cap also reduced the worst burstiness somewhat, which helped `max_segment_interval_s`
- however, the steady-state cadence remained almost unchanged, which confirms that scheduler policy is no longer the main determinant of aggregate throughput at this load

#### `concurrency=8` with `batch_size=2`

Observed metrics:

1. `live_ready` averaged about 6.63 seconds.
2. Average segment interval was about 6.58 seconds.
3. Maximum segment interval was about 7.70 seconds.
4. Total wall time was about 123.0 seconds.
5. All eight sessions completed without failures.

Interpretation:

- startup latency is better than the `concurrency=7` outlier, but steady-state cadence is worse
- this suggests heavily overloaded startup behavior is still sensitive to queue order and timing, even though aggregate throughput degradation remains monotonic
- eight sessions are now clearly a stability-only stress case, not a realistic live serving target

#### Lower-FPS follow-up: `playback_fps=24`, `musetalk_fps=12`, `batch_size=2`

Follow-up run settings:

1. `segment_duration=1.0`
2. `playback_fps=24`
3. `musetalk_fps=12`
4. `batch_size=2`
5. `LIVE_MAX_CONCURRENT_GENERATIONS=9`
6. `HLS_SCHEDULER_MAX_BATCH=20`

Observed metrics:

1. `concurrency=1`: `live_ready=1.51s`, `avg_segment_interval=0.63s`, `max_segment_interval=1.02s`, `wall_time=11.4s`
2. `concurrency=4`: `live_ready=3.38s`, `avg_segment_interval=2.56s`, `max_segment_interval=3.17s`, `wall_time=46.8s`
3. `concurrency=5`: `live_ready=3.65s`, `avg_segment_interval=3.26s`, `max_segment_interval=4.27s`, `wall_time=60.9s`
4. `concurrency=6`: `live_ready=4.25s`, `avg_segment_interval=3.97s`, `max_segment_interval=5.32s`, `wall_time=73.8s`
5. `concurrency=7`: `live_ready=4.88s`, `avg_segment_interval=4.57s`, `max_segment_interval=6.08s`, `wall_time=85.0s`
6. `concurrency=8`: `live_ready=5.31s`, `avg_segment_interval=5.24s`, `max_segment_interval=7.13s`, `wall_time=98.7s`

Comparison against the earlier `playback_fps=30`, `musetalk_fps=15`, `batch_size=2` profile:

1. `concurrency=4` improved from `3.21s` to `2.56s` average segment interval and from `59.2s` to `46.8s` wall time.
2. `concurrency=6` improved from `4.90s` to `3.97s` average segment interval and from `90.3s` to `73.8s` wall time.
3. `concurrency=7` improved from `5.80s` to `4.57s` average segment interval and from `106.7s` to `85.0s` wall time.
4. `concurrency=8` improved from `6.58s` to `5.24s` average segment interval and from `123.0s` to `98.7s` wall time.

Interpretation:

- lowering generation load from `15 fps` to `12 fps` materially improves startup latency, cadence, and total wall time across all tested concurrencies
- the gain is roughly proportional to the reduced per-stream frame demand, which is consistent with the backend still operating near the same aggregate generation ceiling
- this profile feels faster because each stream asks the model for fewer frames per second, not because the underlying shared GPU path suddenly became much faster
- even with the lower-FPS profile, `concurrency=4+` remains throttled for 1-second HLS segments, so the change improves headroom but does not remove the core throughput bottleneck

Important scheduler implication:

- with `HLS_SCHEDULER_MAX_BATCH=20`, `batch_size=2`, and `concurrency=7` or `8`, one scheduler pass only contributes `14` to `16` frames if every job gets a single slice, still leaving some nominal headroom under the shared batch cap
- if throughput remains almost flat while that headroom exists, the next optimization target is likely better shared-batch filling and model throughput, not simply pushing per-stream batch size higher

### March 14 batch allocation and deadlock investigation

This section captures findings from a focused investigation into why `max_combined_batch_size=32` caused complete scheduler deadlock at `concurrency=8`.

#### Discovery: `_memory_bucket` deadlock

The `HLSGPUStreamScheduler._run_generation_batch` method acquires a GPU memory lease via:

```python
with self.manager.gpu_memory.allocate(lease_batch_size):
```

Where `lease_batch_size` comes from `_memory_bucket(total_batch)`. The `GPUMemoryManager` was initialized with `max_concurrent_inferences=5` in `ParallelAvatarManager`, creating an internal semaphore or budget pool with a small fixed capacity.

When `_memory_bucket` was expanded to return values up to 48 or 64 (to match the new `max_combined_batch_size=32`), the allocator tried to lease 32 slots from a pool that had capacity for roughly 5-8. The `allocate()` call blocked forever waiting for slots that would never be released.

Observed symptoms:

1. All 8 jobs queued successfully with `prep=11.48s` each.
2. After the last `queued` log line, the server went completely silent — no UNet logs, no compose logs, no errors.
3. All 8 sessions timed out after 300 seconds with zero segments produced.
4. The scheduler loop was alive but permanently blocked inside `gpu_memory.allocate()`.

#### Root cause

The `_memory_bucket` value is **not the GPU batch size**. It is a lease count against the `GPUMemoryManager`'s internal budget. The scheduler runs one batch at a time in a serial loop — it never needs more than one lease regardless of how many frames are in the batch.

The confusion arose because `_memory_bucket` was treated as if it should scale with `max_combined_batch_size`, but these are fundamentally different concepts:

- `max_combined_batch_size` = frame count for the UNet forward pass (should be large for GPU efficiency)
- `_memory_bucket` = lease count for the memory manager's admission semaphore (should be minimal for the serial scheduler)

#### Fix applied

```python
@staticmethod
def _memory_bucket(batch_size: int) -> int:
    return 1
```

The scheduler is the sole consumer of the GPU in its loop. It processes one batch at a time, then the next. It only ever needs one lease slot. The actual GPU batch size (32 frames into UNet) is controlled by `max_combined_batch_size` and the `_select_jobs_locked` allocation logic, not by the memory lease.

#### Verification

After the fix, all 8 streams completed successfully:

```
concurrency=8, completed=8, failed=0
avg_time_to_live_ready_s=8.333
avg_segment_interval_s=5.209
max_segment_interval_s=5.744
wall_time_s=99.7
```

This confirmed:

1. The deadlock was purely in the memory lease, not in GPU capacity.
2. The GPU can process 8 concurrent streams to completion.
3. The scheduler's serial batch loop is architecturally correct — it does not need concurrent GPU access.

#### NVENC broken pipe at high concurrency

During the 8-stream run, the server logs showed:

```
⚠️ Encoder h264_nvenc failed, retrying with fallback: [Errno 32] Broken pipe
```

The RTX 3090 supports approximately 3-5 concurrent NVENC hardware encoding sessions. With 8 encode workers all attempting h264_nvenc simultaneously, some sessions exceed the hardware limit and fail. The fallback to libx264 (CPU) works but adds latency and CPU contention.

This is a secondary bottleneck that affects encode throughput but is not the primary cause of the 5.2-second segment intervals.

#### `torch.compile` investigation and failure

An attempt was made to add `torch.compile` for UNet and VAE to improve raw GPU throughput by 2-3x. This encountered two blocking issues:

1. **VAE compile**: `torch.compile(self.vae.vae.decoder)` returns an `OptimizedModule`, not an `nn.Module`. Assigning it back to `self.vae.vae.decoder` fails because PyTorch's module system rejects non-Module children.

2. **UNet compile**: After `torch.compile(self.unet.model)`, the result is a callable wrapper. Subsequent code accessing `self.unet.model.dtype` or `next(self.unet.model.parameters())` crashes with `AttributeError: 'function' object has no attribute 'parameters'`.

3. **Double call**: `compile_models()` was called both at the end of `_init_models()` and in `__init__`, causing the error to surface during `_prepare_job` when the scheduler accessed `self.manager.unet.model.dtype`.

Current status:

- `compile_models()` calls are disabled (`MUSETALK_COMPILE=0` by default)
- The `.dtype` reference in `_prepare_job` was hardcoded to `torch.float16` as a safety measure
- A proper `torch.compile` integration requires: saving dtype/parameter references before compilation, compiling the full module (not submodules), and auditing all `.dtype` / `.parameters()` access sites throughout the codebase
- This remains a Phase 2 optimization item

#### Aggregate throughput analysis

The March 14 results at `concurrency=8` with `max_combined_batch_size=32` revealed the fundamental throughput ceiling:

```
Single stream:  wall_time=11.4s, 213 frames → 18.7 fps
Eight streams:  wall_time=99.7s, 213×8=1704 frames → 17.1 fps

Ratio: 17.1 / 18.7 = 0.91x
```

The aggregate GPU throughput is nearly identical at 1 vs 8 streams. The RTX 3090 produces approximately 18-20 fps of MuseTalk output (PE + UNet + VAE combined) regardless of how many streams share the batch.

The 5.2-second segment interval at `concurrency=8` is explained by:

```
18 fps aggregate ÷ 8 streams = 2.25 fps per stream
12 frames per segment ÷ 2.25 fps = 5.3 seconds per segment
```

This matches the observed 5.2 seconds precisely.

#### Implication for target parity

To serve 8 streams at 12fps with 1-second segment cadence matching single-stream speed, the backend would need:

```
8 streams × 12 fps = 96 fps aggregate throughput
Current: ~18-20 fps
Deficit: ~5x
```

This deficit cannot be closed by scheduler improvements alone. The options are:

1. **torch.compile** (2-3x UNet/VAE speedup) — would bring aggregate to ~36-54 fps, enough for 3-4 streams at 12fps but not 8
2. **TensorRT** (5-8x speedup) — would bring aggregate to ~90-160 fps, sufficient for 8 streams
3. **Lower musetalk_fps** — reduces per-stream demand proportionally:
   - At 6fps: 8×6=48 fps needed, achievable with torch.compile
   - At 4fps: 8×4=32 fps needed, achievable with current GPU (barely)
   - At 3fps: 8×3=24 fps needed, comfortably within current GPU capacity
4. **Second GPU** — doubles aggregate throughput to ~36-40 fps
5. **Frame duplication** — generate at low musetalk_fps but display at high playback_fps by repeating frames

### March 14 scheduler batch fill and worker scaling

The following scheduler changes were validated during the March 14 investigation:

#### Round 3 fill guard removal

The `_select_jobs_locked` method previously skipped Round 3 (aggressive fill) when `len(warmed_jobs) > aggressive_fill_max_active_jobs` (default 4). This meant that at `concurrency=8`:

- Round 2 allocated 8 jobs × 2 frames = 16 frames, but capped at `max_combined_batch_size`
- Round 3 was SKIPPED because 8 warmed jobs > 4 threshold
- Result: GPU ran at partial capacity even though more frames could have been batched

The guard was removed so Round 3 always runs, filling remaining GPU capacity across all schedulable jobs regardless of count. This is controlled by setting `HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS=999` at launch.

#### Compose and encode worker auto-scaling

The compose and encode executor pool sizes were changed from fixed values to auto-scaled based on CPU count:

```python
cpu_count = os.cpu_count() or 8
effective_compose = max(6, int(compose_workers), min(10, cpu_count // 2))
effective_encode = max(6, int(encode_workers), min(10, cpu_count // 2))
```

This ensures that at 8 concurrent streams, there are enough CPU workers to handle the increased volume of compose and encode tasks without creating a queue bottleneck.

### Updated performance conclusion

The new HLS scheduler clearly improved:

1. lifecycle safety
2. startup consistency
3. ability to complete concurrent jobs without obvious worker corruption

But the latest measurements now show a more nuanced picture:

1. single-stream cadence is healthy under the tested 1-second segment profile
2. two-stream cadence is close to realtime, but still near the throttle threshold
3. three-stream behavior is much better than before, but still too slow for a clean realtime claim
4. four-, five-, six-, seven-, and eight-stream runs now complete reliably, but they are clearly operating in a heavily throttled regime
5. reducing per-stream `batch_size` from `4` to `2` at `concurrency=4` and `6` does not materially change aggregate throughput under the current shared scheduler settings
6. at `concurrency=7` and above, startup fairness can become unstable without scheduler tuning, but the newer startup-first scheduler materially improves first-live latency and total wall time
7. even after that tuning, steady-state segment cadence remains the dominant bottleneck at high concurrency
8. lowering `musetalk_fps` from `15` to `12` with `playback_fps=24` materially improves high-concurrency behavior, but mainly by reducing per-stream work rather than by changing the backend's aggregate throughput ceiling
9. raising `max_combined_batch_size` from 8/20 to 32 with the `_memory_bucket=1` fix eliminates the scheduler deadlock and allows all 8 streams to complete, but does not increase raw GPU throughput because UNet forward pass time scales sub-linearly with batch size (bs=32 is only ~1.75x slower than bs=8, but processes 4x more frames — net gain in per-tick frames but not enough to close the 5x deficit)
10. the aggregate GPU throughput ceiling of ~18-20 fps is a hardware constant of the RTX 3090 running MuseTalk's PE+UNet+VAE pipeline at float16 precision; only model-level optimizations (torch.compile, TensorRT) or hardware changes can meaningfully increase this

### March 15 follow-up: best current `concurrency=8` result with live GPU metrics

After the later scheduler cleanup, latent tensorization, idle-HLS caching, and GPU instrumentation work, the current best `24/12`, `concurrency=8`, `batch_size=2` run looks like this:

```json
{
  "concurrency": 8,
  "completed": 8,
  "failed": 0,
  "avg_time_to_live_ready_s": 8.035,
  "avg_segment_interval_s": 5.126,
  "max_segment_interval_s": 5.362,
  "wall_time_s": 93.9,
  "gpu": {
    "samples": 92,
    "avg_util_pct": 36.9,
    "peak_util_pct": 100.0,
    "avg_memory_used_mb": 11389.3,
    "peak_memory_used_mb": 12237.0,
    "avg_memory_util_pct": 14.36,
    "peak_memory_util_pct": 63.0
  }
}
```

Interpretation:

1. this is a real improvement over the earlier `24/12`, `concurrency=8` runs that were landing closer to `99-118s` wall time with worse tail jitter
2. the largest win is reduced worst-case cadence jitter (`max_segment_interval_s` improved materially)
3. `avg_segment_interval_s=5.126` implies effective aggregate generation throughput of roughly `96 / 5.126 = 18.7 fps`
4. that is slightly better than the earlier `~18.3 fps`, but it is still inside the same overall `18-20 fps` hardware ceiling band
5. the result is therefore best interpreted as an overhead/jitter reduction, not a step-change in raw model throughput

GPU interpretation:

1. `peak_util_pct=100` confirms the GPU still saturates during important parts of the run
2. `avg_util_pct=36.9` does not mean the GPU is not the bottleneck; the average includes prep, encode, manifest updates, and idle gaps between compute bursts
3. `peak_memory_used_mb=12237` leaves large VRAM headroom on a 24GB card, so VRAM remains a secondary limit here
4. low average VRAM-controller utilization also supports the conclusion that the system is compute-bursty, not memory-capacity-bound

### `batch_size=2` vs `batch_size=4` at `concurrency=8`

The latest controlled comparison continues to show that `batch_size` is no longer a meaningful lever at high concurrency under the current scheduler:

```json
{
  "batch_size": 2,
  "avg_time_to_live_ready_s": 8.035,
  "avg_segment_interval_s": 5.126,
  "max_segment_interval_s": 5.362,
  "wall_time_s": 93.9
}
```

```json
{
  "batch_size": 4,
  "avg_time_to_live_ready_s": 8.163,
  "avg_segment_interval_s": 5.108,
  "max_segment_interval_s": 5.424,
  "wall_time_s": 94.0
}
```

Interpretation:

1. `avg_segment_interval_s` is effectively flat
2. startup is slightly worse at `batch_size=4`
3. `wall_time_s` is identical within normal run-to-run noise
4. the scheduler is already assembling enough total work per GPU turn that larger per-stream batches do not unlock a new throughput regime

Conclusion:

- keep `batch_size=2` as the default HLS setting for now
- do not expect `batch_size=4` to materially improve `concurrency=8` throughput
- focus future effort on model-path acceleration, encode-path cleanup, and startup-path overhead instead

### Startup latency vs steady-state cadence

One important lesson from the March 15 runs is that startup latency and steady-state cadence are different bottlenecks.

Example older run at `concurrency=5`:

```json
{
  "concurrency": 5,
  "avg_time_to_live_ready_s": 6.097,
  "avg_segment_interval_s": 3.222,
  "max_segment_interval_s": 4.154,
  "wall_time_s": 60.3,
  "gpu": {
    "avg_util_pct": 40.83,
    "peak_util_pct": 100.0,
    "avg_memory_used_mb": 10818.6,
    "peak_memory_used_mb": 12520.0
  }
}
```

What this means:

1. startup delay rises with concurrency before VRAM becomes a problem
2. a moderate average GPU utilization does not imply the startup path is healthy
3. `peak_util_pct=100` together with modest average utilization is the signature of a bursty pipeline: prep, GPU batch, compose, encode, wait, repeat
4. the browser cannot begin playback until the first live chunk exists, so prep time, scheduler queue time, and first-chunk encode time all directly inflate `avg_time_to_live_ready_s`

The practical diagnostic rule is:

1. high `avg_time_to_live_ready_s` plus moderate average GPU usage usually means startup-pipeline contention
2. high `avg_segment_interval_s` with 100% utilization peaks usually means shared compute saturation
3. high memory usage near the device limit would indicate actual VRAM pressure, but that is not what the current runs show

### HLS tuning cheat sheet

These are the HLS env vars that have been actively tuned during the March 2026 investigation, along with what they actually do in the current codebase.

| Env var | Current meaning | Real impact on throughput | Current recommendation |
|---|---|---|---|
| `MUSETALK_COMPILE=1` | Enables `torch.compile` warmup for UNet + VAE in `avatar_manager_parallel.py` | Potentially the only knob here that can raise the raw model-throughput ceiling | Worth benchmarking, but treat as environment-sensitive and validate on the exact runtime stack |
| `HLS_SCHEDULER_MAX_BATCH=32` | Caps the total combined frame batch the shared HLS scheduler can process in one GPU turn | Real throughput/fairness lever; too low wastes GPU opportunity, too high increases per-turn latency and jitter | `20-32` is the practical range; `32` is a sensible upper bound for the current scheduler |
| `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4` | Controls the first-round allocation for startup jobs | Limited by each stream's `batch_size`, so it does little when `batch_size=2` | Leave at `2`; values above the stream `batch_size` do not help |
| `HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS=999` | Historical guard controlling whether the scheduler refilled remaining capacity | No longer materially affects behavior because the current scheduler always fills remaining capacity | Ignore for now; effectively a no-op |
| `HLS_COMPOSE_WORKERS=8` | Thread pool size for CPU-side frame composition | Can reduce compose queueing, but does not change the model-throughput ceiling | Use only if compose queue wait is elevated; `6-8` is reasonable, more can oversubscribe CPU |
| `HLS_ENCODE_WORKERS=8` | Thread pool size for encode submission work | Can reduce encode queueing, but can also increase ffmpeg / NVENC contention | Use only if encode queue wait is elevated; `6-8` is reasonable, but not a primary throughput lever |
| `HLS_MAX_PENDING_JOBS=24` | Caps how many HLS jobs can be preparing or queued before rejecting new starts | Does not improve throughput; only changes backlog behavior | Treat as a queueing/admission knob, not a speed knob |

Important caveats:

1. `HLS_SCHEDULER_MAX_BATCH` is a combined frame-count cap, not GB of VRAM
2. `HLS_SCHEDULER_STARTUP_SLICE_SIZE` is capped by per-stream `batch_size`, so it cannot force a `batch_size=2` stream to take `4` frames in the first round
3. `HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS` is still exposed in logs/stats, but the current scheduling loop no longer uses it to block fill behavior
4. `HLS_COMPOSE_WORKERS` and `HLS_ENCODE_WORKERS` are queue-management knobs; they are not substitutes for raising the actual PE + UNet + VAE throughput ceiling

Recommended baseline startup command for current testing:

```bash
export HLS_SCHEDULER_MAX_BATCH=32
export HLS_COMPOSE_WORKERS=6
export HLS_ENCODE_WORKERS=6
python api_server.py --host 0.0.0.0 --port 8000
```

Recommended compile test variant:

```bash
export MUSETALK_COMPILE=1
export HLS_SCHEDULER_MAX_BATCH=32
python api_server.py --host 0.0.0.0 --port 8000
```

Do not expect the following to materially improve current `24/12`, `concurrency=8`, `batch_size=2` throughput by themselves:

- `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4`
- `HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS=999`
- larger pending-job caps

### Final investigative conclusion

The current system has two different capacity limits:

1. memory capacity
2. realtime throughput capacity

The current implementation was tuned mostly around memory capacity.

The load tests proved that realtime throughput capacity is the stricter limit.

## Current Root Causes

### 1. HLS delete does not cancel live work

Original behavior:

- `DELETE /hls/sessions/{session_id}` calls `hls_session_manager.delete_session(session_id)`.
- `delete_session()` removes the session object and deletes the output directory.
- The inference worker started by `/hls/sessions/{session_id}/stream` keeps running in the executor.

Impact before the fix:

- Old jobs continue running after the user thinks they are gone.
- GPU logical usage remains allocated.
- The next load test starts with stale work already active.

Current status:

- materially improved
- delete now requests cancellation first and can defer final deletion until the stream has unwound

Relevant files:

- `api_server.py`
- `scripts/hls_session_manager.py`

### 2. Threads are not safely killable

The system uses `ThreadPoolExecutor`.

Important consequence:

- `future.cancel()` only cancels a task if it has not started yet.
- Once a worker is already running inside Python code, there is no safe hard-kill mechanism for that thread.

Impact:

- A "worker cleaner" cannot reliably stop a hung job if the architecture remains thread-only.

Relevant file:

- `scripts/avatar_manager_parallel.py`

### 3. GPU budget is tied to worker exit

GPU budget release happens only when the allocation context exits.

Impact:

- If the worker does not exit, the logical GPU budget remains held.
- Resetting the budget counter manually would be wrong, because the old worker might still be running on the GPU.

Current status:

- partially mitigated by `_memory_bucket` returning 1, so the scheduler only ever holds one minimal lease
- the HLS scheduler's serial loop means the lease is released after each batch completes, not held for the entire stream duration

Relevant file:

- `scripts/concurrent_gpu_manager.py`

### 4. Same-avatar load race

`_get_or_load_avatar()` checks the cache, and if there is a miss, multiple concurrent requests can all load the same avatar independently.

Impact:

- Duplicate disk I/O.
- Slow cold-start behavior.
- More memory pressure.
- More variance during load tests.

Current status:

- fixed with per-avatar load locks

Relevant file:

- `scripts/avatar_manager_parallel.py`

### 5. Shared temp directory per avatar

Original behavior:

- `tmp_dir = f"{self.avatar_path}/tmp"`

That path is shared by all concurrent requests for the same avatar.

Impact before the fix:

- One request can delete temp files that another request still expects.
- Cleanup and partial writes can race.

Current status:

- fixed for streaming paths by request-scoped scratch directories

Relevant file:

- `scripts/api_avatar.py`

### 6. Incomplete request registry and observability

`manager.active_requests` is populated, but HLS and WebRTC requests are not fully treated as first-class lifecycle-managed jobs.

Impact:

- Hard to know which work is stale.
- Hard to cancel by session id.
- Hard to expose progress, last heartbeat, stall detection, or cleanup state.

Current status:

- improved for HLS
- HLS request lifecycle is now more explicit and HLS stats expose scheduler state
- still incomplete as a unified cross-protocol request registry

Relevant files:

- `api_server.py`
- `scripts/avatar_manager_parallel.py`
- `scripts/hls_gpu_scheduler.py`

### 7. `_memory_bucket` returning large values causes scheduler deadlock

Original behavior:

- `_memory_bucket(batch_size)` returned values up to 48 or 64 to match GPU batch capacity.
- `GPUMemoryManager.allocate(lease)` blocked until `lease` slots were available.
- The memory manager had a fixed pool of roughly 5 slots (from `max_concurrent_inferences=5`).
- When `_memory_bucket(32)` returned 32, the allocator blocked forever.

Impact:

- Complete scheduler deadlock at any concurrency where `total_batch > pool_capacity`.
- All streams timed out with zero segments produced.
- No error was logged because the blocking happened inside a `with` statement before any generation code executed.

Current status:

- fixed by returning `1` from `_memory_bucket`
- the scheduler's serial loop only needs one lease at a time

Relevant files:

- `scripts/hls_gpu_scheduler.py`
- `scripts/concurrent_gpu_manager.py`

### 8. NVENC concurrent session limit causes encode failures

The RTX 3090 supports approximately 3-5 concurrent NVENC hardware encoding sessions. With 8 encode workers each spawning a new ffmpeg process per chunk with `h264_nvenc`, some sessions exceed the hardware limit and receive a broken pipe error.

Impact:

- Chunk encoding falls back to `libx264` (CPU), which is slower and adds CPU contention.
- The fallback works correctly but increases per-chunk encode time.
- At 8 concurrent streams × 18 chunks each = 144 ffmpeg process spawns, many of which fail on first attempt.

Potential fix:

- Implement a persistent per-stream ffmpeg encoder that keeps one ffmpeg process alive for the entire generation, eliminating per-chunk process spawning and avoiding the NVENC session limit entirely by using `libx264 -preset ultrafast` in the long-running pipe.

Relevant files:

- `scripts/api_avatar.py` (chunk creation)
- `scripts/hls_gpu_scheduler.py` (encode dispatch)

### 9. `torch.compile` remains environment-sensitive, but the integration is safer now

The original `torch.compile` attempt was broken for three concrete reasons:

1. `torch.compile(module)` returns a callable wrapper, not a normal `nn.Module`
2. code throughout the pipeline accessed `.dtype` and `.parameters()` on the compiled object
3. `compile_models()` was called twice

Those structural issues have now been addressed:

1. dtype values are stored separately before compilation
2. the duplicate compile call has been removed
3. the HLS path uses saved dtype references instead of asking the compiled wrapper for them
4. warmup failures restore the original eager models instead of leaving broken compiled modules active
5. compiled execution now uses `torch.no_grad()` instead of `torch.inference_mode()` in the HLS generation path

Current status:

- the compile path no longer fails in an unsafe way
- failed warmup should now fall back to eager execution rather than crashing the first live request
- however, the current PyTorch + diffusers stack is still sensitive, and warmup failures such as `Inference tensors do not track version counter` remain possible on some environments

Impact:

- `torch.compile` is no longer a correctness blocker
- it is still not a guaranteed throughput win on every machine
- it remains the most promising software-only path to increase the `18-20 fps` ceiling, but it must be validated on the exact target runtime stack

Relevant files:

- `scripts/avatar_manager_parallel.py`
- `scripts/hls_gpu_scheduler.py`

## What This Is Not

This is not mainly:

- a Python GC tuning issue
- a `torch.cuda.empty_cache()` issue
- a 3090-specific bug

A faster GPU may reduce latency, but it will not fix orphaned workers, shared temp paths, or missing cancellation.

It is also not correct to treat "CUDA-level behavior" as meaning "VRAM must be the bottleneck". In the current pipeline, the more important distinction is:

- VRAM controls how much work can fit
- throughput controls how fast the work actually completes

This backend is now clearly throughput-bound before it is VRAM-bound.

## Throughput Findings: Why Low VRAM Usage Still Performs Poorly

The working assumption for much of the tuning effort was:

- if VRAM use is low, there should still be room for more concurrency

The single-stream results disproved that assumption.

### Key principle

VRAM usage is a capacity metric. It is not a speed metric.

A request can use only a fraction of available VRAM and still be too slow for realtime streaming because of:

1. GPU compute throughput limits
2. CPU preprocessing cost
3. CPU encode cost
4. kernel launch overhead
5. synchronization overhead
6. small-batch inefficiency
7. file I/O and manifest update overhead

### The hardware throughput ceiling

The RTX 3090 produces approximately 18-20 frames per second of MuseTalk output (PE + UNet + VAE combined at float16 precision). This is a hardware constant under the current model architecture and has been validated empirically:

```
Single stream:  213 frames / 11.4s wall = 18.7 fps
Eight streams:  1704 frames / 99.7s wall = 17.1 fps
Ratio: 0.91x — nearly identical aggregate throughput
```

Updated March 15 reference point:

```
Eight streams (improved run): 1704 frames / 93.9s wall = 18.1 fps
Effective cadence-based estimate: 96 demanded fps / 5.126s interval = 18.7 fps
```

Latest March 15 post-compose / PE refactor reference point:

```json
{
  "concurrency": 8,
  "avg_time_to_live_ready_s": 3.282,
  "avg_segment_interval_s": 2.301,
  "max_segment_interval_s": 3.521,
  "wall_time_s": 44.4,
  "gpu": {
    "avg_util_pct": 75.18,
    "peak_util_pct": 100.0,
    "avg_memory_used_mb": 8998.2,
    "peak_memory_used_mb": 10588.0
  }
}
```

Representative scheduler-finished lines from that same run:

```text
avg_gpu_batch ~= 0.73s to 0.81s
avg_compose ~= 0.12s to 0.14s
avg_encode ~= 0.47s to 0.68s
first_chunk ~= 2.04s to 3.67s
post_gen_drain ~= 0.64s to 3.56s
```

This is the first run in the current HLS architecture that clearly breaks out of the older `~5.1-5.2s` interval band. The compose rewrite and CPU-side PE precompute materially improved both startup and steady-state cadence on the RTX 3090.

This reinforces the same conclusion:

- the system can still be improved around the edges
- however, the old `18-20 fps` aggregate ceiling estimate is now clearly stale for the current code path and must be re-measured from the new baseline before being treated as authoritative

The batch size affects per-tick latency but not aggregate throughput in a meaningful way:

```
UNet at bs=8:   ~20ms (8 frames / 20ms = 400 fps equivalent, but other stages dominate)
UNet at bs=16:  ~25ms (16 frames / 25ms = 640 fps equivalent)
UNet at bs=32:  ~35ms (32 frames / 35ms = 914 fps equivalent)

But the full pipeline (PE + UNet + VAE + compose + encode) adds ~20-30ms overhead per tick,
so actual per-tick throughput is:
  bs=8:   8 frames / 45ms = 178 fps
  bs=16:  16 frames / 55ms = 291 fps
  bs=32:  32 frames / 65ms = 492 fps
```

Even at the most optimistic bs=32 throughput, serving 8 streams at 12fps (96 fps needed) would require segment intervals of 96/492 × 8 ≈ 1.56 seconds — still above the 1-second target.

### Relevant bottlenecks in this repository

#### 1. CPU audio preprocessing still matters

The pipeline does not begin purely on the GPU.

Audio loading and feature preparation still include CPU-side work before the main GPU generation path contributes live output.

#### 2. Whisper and generation are compute-bound, not memory-bound

Once audio reaches whisper encoding and the main model forward path, the GPU can be saturated even while overall VRAM use remains modest.

This is normal:

- low VRAM use with high GPU utilization means compute is the bottleneck, not memory footprint.

#### 3. Small live batches are not automatically efficient

The current HLS path uses small per-request live batches.

That causes:

1. more kernel launch overhead per unit of useful work
2. less effective GPU utilization than a well-batched scheduler
3. more overhead when multiple Python threads all try to drive the same shared model stack

#### 4. HLS segment encoding is a separate bottleneck

Chunk creation is not free.

Even after frame generation, every chunk still has to be encoded into a `.ts` segment. In the current implementation this is still part of the end-to-end latency budget for each live stream.

The current per-chunk ffmpeg spawning pattern creates additional overhead:

- 8 streams × 18 chunks = 144 ffmpeg process spawns
- Each spawn: fork (~5-10ms) + NVENC attempt → broken pipe → libx264 fallback retry
- A persistent per-stream ffmpeg pipe would eliminate this overhead entirely

#### 5. One-second segments create a hard realtime requirement

With `segment_duration=1.0`, the backend must sustain roughly one second of useful output per second of wall time, plus startup overhead, plus encode overhead, plus manifest churn.

If actual segment cadence is around 3.2 seconds at `concurrency=1`, the system is already about 3x slower than the realtime target.

### What to be mindful of for GPU AI inferencing

When evaluating single-GPU inference systems, focus on these principles:

1. Memory footprint and throughput are different constraints.
2. Small concurrent jobs on one GPU often interfere with each other instead of scaling linearly.
3. Multiple Python threads do not give you direct, efficient "CUDA core management".
4. Shared models need an explicit scheduler, not blind thread-level concurrency.
5. Real-time streaming must be judged against wall-clock output cadence, not just successful completion.
6. CPU stages around the model can dominate end-to-end latency even when the GPU is busy.
7. A GPU at 100 percent utilization is not proof of efficiency. It only proves the device is busy.
8. The correct optimization target is stable time-to-first-chunk and stable chunk cadence.

### What this means operationally

To support more users safely, the engineering problem is not:

- "how do we force more simultaneous CUDA threads onto the GPU?"

The real problem is:

- "how do we schedule, batch, and bound work so the GPU produces live chunks fast enough?"

That requires scheduler design, admission control, and profiling. It is not solved by raw multithreading.

## Recommended End-State Architecture

The most robust solution is to separate the HTTP API server from the GPU execution engine.

### Recommended topology

1. API server process
2. GPU worker service process per GPU
3. Supervisor around the GPU worker process
4. Optional separate encoder subprocesses per request

### Why this is the right design

The API server should not directly own long-running CUDA work. The server should only:

- validate requests
- create sessions
- submit jobs
- poll job state
- cancel jobs
- serve manifests and segments

The GPU worker service should:

- load models once
- own the CUDA context
- run the scheduling logic
- expose cooperative cancellation
- emit heartbeats and progress

The supervisor should:

- monitor GPU worker health
- detect stalled worker conditions
- restart the GPU worker process if needed
- fail active jobs cleanly during restart

This architecture gives you a real recovery story. If the GPU worker gets wedged, you can restart that process without bringing down the whole HTTP server.

## Target Design Principles

### 1. Every live request must have a real lifecycle

Each request should have:

- `request_id`
- `session_id`
- `stream_type` (`hls`, `session_stream`, `webrtc`)
- `status`
- `created_at`
- `started_at`
- `finished_at`
- `last_progress_at`
- `cancel_requested_at`
- `terminal_reason`
- `worker_owner`

### 2. Cancellation must be cooperative first

Every long-running generation path must check for cancellation:

- before audio feature extraction
- before whisper chunk generation
- after whisper chunk generation
- before each generation batch
- after each generation batch
- before each chunk encode
- after each chunk encode

### 3. Hard kill must be process-based

If cooperative cancellation fails:

- kill the encoder subprocess if encoding is stuck
- if the GPU worker itself is stalled, restart the GPU worker process

Do not attempt hard cleanup by mutating counters or deleting files while compute is still active.

### 4. Admission control must be explicit

Do not silently accept more work than the GPU can sustain.

Each live stream request should either:

- start immediately
- enter a bounded queue with timeout
- fail fast with a backpressure response

### 5. Per-request scratch space only

Never use a shared temp path for concurrent live requests of the same avatar.

Each request must have its own scratch directory, such as:

- `results/.../scratch/{request_id}/`

### 6. Backends must be measurable

The system must report:

- active jobs
- queued jobs
- cancelled jobs
- orphan cleanup count
- average time in queue
- average time to first chunk
- chunk encode latency
- GPU budget in use
- per-request last progress timestamp

## Robust Solution Design

## A. Introduce a request registry

Create a single request registry used by HLS, SSE, and WebRTC.

Suggested new file:

- `scripts/request_registry.py`

Responsibilities:

1. Register request metadata.
2. Link request to session and stream type.
3. Hold cancel handles.
4. Record progress heartbeats.
5. Mark terminal status.
6. Reap completed requests.

Suggested request states:

- `queued`
- `starting`
- `running`
- `cancelling`
- `completed`
- `cancelled`
- `failed`
- `timed_out`
- `orphaned`

## B. Add cancellation to sessions

Extend session models so each session knows its active request and cancellation state.

Files to update:

- `scripts/hls_session_manager.py`
- `scripts/session_manager.py`
- `scripts/webrtc_manager.py`

Suggested new session fields:

- `active_request_id`
- `cancel_requested`
- `cancel_requested_at`
- `deleting`
- `terminal_reason`

Deletion flow should become:

1. Mark session as deleting.
2. If it has an active request, request cancellation.
3. Wait for request termination or timeout.
4. Only then remove files and session metadata.

## C. Make `inference_streaming()` cancellable

Update `APIAvatar.inference_streaming()` to accept:

- `cancel_event`
- `progress_callback`
- `scratch_dir`

Files to update:

- `scripts/api_avatar.py`

Specific changes:

1. Stop using `self.avatar_path/tmp` for live requests.
2. Accept a request-specific `scratch_dir`.
3. Use `try/finally` cleanup for scratch dir.
4. Emit progress callbacks at major stages.
5. Raise a distinct cancellation exception when `cancel_event` is set.

Suggested progress events:

- `audio_features_started`
- `audio_features_done`
- `whisper_started`
- `whisper_done`
- `generation_started`
- `batch_done`
- `chunk_started`
- `chunk_done`
- `stream_finished`

## D. Add per-avatar load locks

Fix same-avatar cold-load duplication.

Files to update:

- `scripts/avatar_manager_parallel.py`

Recommended change:

- Maintain `avatar_load_locks: Dict[str, threading.Lock]`
- Lock the entire cache-miss load path per `avatar_id`

Result:

- only one request loads `test_avatar` from disk
- all others wait and then get the cached instance

Current status:

- implemented

## E. Replace shared executor ownership with a GPU worker service

This is the most robust structural improvement.

Suggested new files:

- `scripts/gpu_worker_service.py`
- `scripts/gpu_worker_supervisor.py`
- `scripts/gpu_worker_protocol.py`

### GPU worker service responsibilities

1. Load models once.
2. Own the CUDA context.
3. Receive jobs from the API server over IPC.
4. Maintain active job state.
5. Enforce concurrency and queue policy.
6. Support cancellation.
7. Emit progress and heartbeats.

### Supervisor responsibilities

1. Start the GPU worker process.
2. Monitor heartbeat.
3. Detect stalls.
4. Restart the GPU worker on fatal wedge.
5. Mark in-flight jobs as failed or cancelled after restart.

### IPC options

Preferred:

- local multiprocessing queues or a Unix domain socket

Alternative:

- Redis or NATS if you want multi-host scaling later

### Why not per-request model processes

Do not load the full models separately in one process per stream on a 24 GB GPU. That duplicates VRAM and defeats throughput.

The correct process split is:

- one API process
- one long-lived GPU worker process per GPU

## F. Isolate ffmpeg work

Chunk encoding is already an external subprocess. Keep it that way and make it part of request cancellation.

Files to update:

- `scripts/api_avatar.py`

Recommended changes:

1. Track child process PID per request.
2. On cancel, terminate ffmpeg subprocess first.
3. If it does not exit, kill it.
4. Report encoder timeout separately from GPU inference timeout.

Optional improvement:

- support NVENC for HLS chunk encoding where acceptable

Notes:

- This helps CPU load and chunk latency.
- It does not fix the orphan-worker problem by itself.

Additional recommendation:

- implement a persistent per-stream ffmpeg encoder that keeps one ffmpeg process alive per HLS stream for the entire generation
- this eliminates the per-chunk process spawn overhead (144 forks at 8 streams) and avoids the NVENC session limit by using a single long-running `libx264 -preset ultrafast` pipe per stream
- segments are produced by ffmpeg's built-in HLS muxer rather than by individual chunk creation calls

## G. Replace the current GPU budget loop with admission control

The current `GPUMemoryManager` is a logical accounting helper. It should become part of an explicit admission controller instead of a spin-wait loop.

Files to update:

- `scripts/concurrent_gpu_manager.py`

Recommended changes:

1. Track allocations by `request_id`.
2. Replace anonymous budget usage with named leases.
3. Expose waiters, active leases, and queue depth.
4. Remove `torch.cuda.synchronize()` from the wait loop.
5. Use condition variables or semaphores.

Recommended behavior:

- if capacity exists, lease immediately
- if queue enabled, wait with timeout
- otherwise reject with backpressure

Current note:

- The HLS scheduler's `_memory_bucket` returning `1` means the memory manager is effectively bypassed for HLS workloads. The scheduler only holds one lease at a time in its serial loop. However, the SSE and non-HLS inference paths still use the full memory manager, so these improvements remain relevant for those codepaths.

## H. Separate "session delete" from "job cancel"

Add explicit cancel endpoints.

Suggested endpoints:

- `POST /hls/sessions/{session_id}/cancel`
- `POST /sessions/{session_id}/cancel`
- `POST /webrtc/sessions/{session_id}/cancel`

Delete should mean:

- remove the session after work is already stopped or after timeout cleanup

Cancel should mean:

- stop the active work but keep session metadata around long enough to report the outcome

This makes operational behavior much easier to reason about.

## I. Add watchdogs and health endpoints

Add watchdog checks for:

1. request heartbeat stale
2. no chunk emitted for too long
3. no batch progress for too long
4. worker heartbeat stale

Suggested new endpoints:

- `GET /stats/requests`
- `GET /stats/requests/{request_id}`
- `GET /stats/worker`
- `GET /stats/queue`

## J. Add startup warm paths

For predictable latency:

1. Warm the avatar cache for heavily used avatars.
2. Warm whisper encoder path once on startup.
3. Warm chunk encode path with a tiny dummy segment if needed.

This reduces first-user penalties and removes cold-start skew from load tests.

## Implementation Plan

## Phase 0: Immediate safety fixes

Goal:

- stop the current contamination between load test runs

Files:

- `api_server.py`
- `scripts/hls_session_manager.py`
- `scripts/avatar_manager_parallel.py`
- `scripts/api_avatar.py`

Changes:

1. Add a request registry entry for every HLS stream.
2. Add per-request cancel event.
3. Make HLS delete request cancellation first, then wait briefly, then delete files.
4. Add per-avatar load locks.
5. Replace shared temp dir with request-specific scratch dir.
6. Ensure HLS worker removes itself from `manager.active_requests` on completion.
7. Record `last_progress_at` during streaming.

Expected result:

- stale workers no longer survive normal delete flow
- repeated load tests become comparable

Status:

- completed

Implemented during this investigation:

1. cancel-before-delete behavior for HLS
2. active request removal on worker completion
3. cooperative cancellation checks in `inference_streaming()`
4. per-avatar load locks
5. improved progress logging
6. request-scoped scratch directories for streaming
7. stronger HLS request reaping through completion callbacks

Still remaining in Phase 0:

1. unify lifecycle tracking further across HLS and SSE
2. add richer per-request observability and timing data

## Phase 0.5: Single-stream profiling and realtime target validation

Goal:

- prove where end-to-end time is going before increasing concurrency further

Why this phase is required:

- the system is already slower than realtime at `concurrency=1`
- concurrency tuning without single-stream profiling will misdiagnose the bottleneck
- the new scheduler improved completion behavior, but the current cadence numbers still show a large gap between functional completion and realtime viability

Files:

- `scripts/api_avatar.py`
- `api_server.py`
- `load_test.py`

Changes:

1. Record time spent in:
   - audio loading
   - audio feature extraction
   - whisper chunk creation
   - per-batch generation
   - frame post-processing
   - segment encode
   - playlist update
2. Expose per-request timing breakdown in status or stats endpoints.
3. Update load testing to capture:
   - time to first generated frame
   - time to first chunk
   - average chunk encode time
   - average batch time
4. Define a strict single-stream acceptance target:
   - average chunk interval near segment duration
   - predictable time-to-live-ready
   - no multi-second drift by the middle of a stream

Expected result:

- clear attribution of the slow path
- a defensible basis for later scheduler and admission-control work

Status:

- partially completed
- the HLS scheduler now records per-stage timing (PE, UNet, VAE, compose, encode) in finalization logs
- March 14 measurements confirmed the aggregate GPU throughput ceiling of ~18-20 fps

## Phase 1: Robust monolith hardening

Goal:

- make the current single-process architecture stable enough for realistic staging load tests

Files:

- `api_server.py`
- `scripts/request_registry.py`
- `scripts/concurrent_gpu_manager.py`
- `scripts/api_avatar.py`
- `load_test.py`

Changes:

1. Introduce a real request registry.
2. Add cancel endpoints.
3. Add explicit request states.
4. Add queue depth and active lease stats.
5. Add request heartbeat and stall thresholds.
6. Improve load test to record:
   - queue wait time
   - live ready latency
   - chunk interval
   - server cancellation outcome
7. Add an explicit cap on simultaneous HLS live generations.
8. Reject or queue excess live HLS starts instead of relying only on VRAM accounting.
9. Move all request scratch work to request-specific directories.

Status:

- partially completed

Implemented during this investigation:

1. SSE live generation now has explicit compute-slot backpressure.
2. HLS no longer uses one executor thread per live stream.
3. HLS now uses a shared GPU scheduler with:
   - one scheduler thread
   - a prep worker pool
   - a separate encode worker pool
4. HLS starts are now admitted through a bounded scheduler queue instead of direct GPU-thread fan-out.
5. HLS stats now expose scheduler queue depth, per-job progress, pending encodes, and last-progress age.
6. HLS session state now explicitly progresses through `preparing`, `queued`, `generating`, and `streaming`.
7. Request scratch work is isolated per stream.
8. `_memory_bucket` returns 1, preventing scheduler deadlock.
9. Round 3 fill guard removed for better GPU utilization at high concurrency.
10. Compose and encode workers auto-scale based on CPU count.

Expected result:

- better visibility
- deterministic behavior under cancellation
- clear capacity limits

Important note:

- based on current measurements, memory admission and throughput admission must remain separate controls
- the new HLS scheduler is the correct architectural direction, but it still needs measurement and tuning before claiming safe two-stream realtime capacity

## Phase 1.5: Startup-path optimization

Goal:

- reduce `avg_time_to_live_ready_s` under realistic concurrent HLS starts without making steady-state throughput worse

Why this phase is required:

- current high-concurrency startup delay is coming from prep contention, scheduler queueing, and first-chunk compose/encode
- synchronized starts amplify this, but even staggered real-user arrivals still pay the same startup pipeline cost
- raising thread counts alone is unlikely to solve this and may simply create more contention

Files:

- `scripts/hls_gpu_scheduler.py`
- `scripts/api_avatar.py`
- `scripts/hls_session_manager.py`
- `scripts/avatar_manager_parallel.py`
- `load_test.py`

Planned changes:

1. **Partial prep for first-live only**
   - prepare only enough audio/Whisper chunks for the first `1-2` HLS segments up front
   - enqueue the stream as soon as first-live data is ready
   - continue preparing the remaining audio/chunks in the background

2. **Explicit first-chunk priority**
   - give new streams a dedicated path to reach first generated chunk quickly
   - prioritize first compose/encode work over later chunks from already-live streams

3. **Warm paths for common avatars**
   - keep common avatars warm in cache
   - keep idle HLS assets cached
   - optionally maintain a small pool of pre-created HLS sessions for common demos / test avatars

4. **Adaptive prep admission**
   - do not let prep jobs stampede the same shared resources when the live scheduler is already saturated
   - gate new prep based on current live load, not just queue depth

5. **Separate prep compute if needed**
   - if startup still harms active streams, consider moving Whisper prep to CPU or a separate GPU path
   - this is a tradeoff: slightly slower isolated prep may still produce better user experience overall if it stops harming already-live sessions

6. **Improve startup benchmarking**
   - continue using burst tests, but add staggered and mid-call injection modes as first-class startup benchmarks
   - evaluate startup separately from steady-state cadence

What not to do first:

1. blindly raise `HLS_PREP_WORKERS`
2. keep increasing compose/encode workers without queue evidence
3. expect `batch_size` changes to solve startup delay

Expected result:

- substantially faster time-to-first-video under concurrent starts
- less competition between "new stream startup" and "existing stream steady-state"
- a more realistic match to actual user traffic, where arrivals are staggered rather than perfectly synchronized

## Phase 1.5: Model-level throughput improvements

Goal:

- increase the aggregate GPU throughput ceiling from ~18-20 fps to ~36-54 fps using `torch.compile`, enabling 3-4 concurrent streams at 12fps or 6-8 streams at 6fps

Files:

- `scripts/avatar_manager_parallel.py`
- `scripts/hls_gpu_scheduler.py`

Changes:

1. Save `unet_dtype` and `vae_dtype` as instance attributes before compilation.
2. Replace all `.dtype` access on compiled models with saved attributes:
   - `hls_gpu_scheduler.py` `_prepare_job`: `weight_dtype = self.manager.unet.model.dtype` → `weight_dtype = torch.float16`
   - `hls_gpu_scheduler.py` `_run_generation_batch`: `target_dtype = audio_inputs.dtype` (already fixed)
3. Compile full modules, not submodules:
   - `torch.compile(self.unet.model)` ✓
   - `torch.compile(self.vae.vae)` (not `self.vae.vae.decoder`)
4. Add batch padding to prevent recompilation:
   ```python
   FIXED_SIZES = [4, 8, 16, 32]
   padded_batch = next(s for s in FIXED_SIZES if s >= actual_batch)
   # Pad inputs, run forward pass, trim outputs
   ```
5. Warmup all expected batch sizes during startup (adds 2-5 minutes to boot).
6. Call `compile_models()` exactly once, after `_init_models()`, gated by `MUSETALK_COMPILE=1`.

Expected result:

- 2-3x throughput improvement on UNet + VAE
- aggregate throughput of ~36-54 fps
- sufficient for 3-4 concurrent streams at 12fps with 1-second segment cadence
- sufficient for 6-8 concurrent streams at 6fps

Prerequisite:

- PyTorch >= 2.0
- verify `torch.compile` availability on deployment target

## Phase 2: API server and GPU worker split

Goal:

- isolate HTTP lifecycle from CUDA lifecycle

Files:

- new `scripts/gpu_worker_service.py`
- new `scripts/gpu_worker_supervisor.py`
- new `scripts/gpu_worker_protocol.py`
- `api_server.py`

Changes:

1. Move model loading out of the API server.
2. Move inference scheduling into a dedicated GPU worker process.
3. API server submits start, cancel, and status messages over IPC.
4. GPU worker owns request progress and cancellation.
5. Supervisor restarts GPU worker on heartbeat failure.

Expected result:

- API remains responsive even if GPU work wedges
- real recovery path exists

## Phase 2.5: Persistent encoder integration

Goal:

- eliminate per-chunk ffmpeg process spawning and NVENC session limit errors

Files:

- new `scripts/hls_persistent_encoder.py`
- `scripts/hls_gpu_scheduler.py`

Changes:

1. Create `PersistentHLSEncoder` class that keeps one ffmpeg process alive per stream.
2. Accept raw BGR frames on stdin, output HLS TS segments via ffmpeg's `-f hls` muxer.
3. Use `libx264 -preset ultrafast` in the persistent pipe to avoid NVENC session limits.
4. Create encoder in `_prepare_job`, pass to `HLSStreamJob`.
5. Change `_dispatch_encode` to call `persistent_encoder.submit_frames()` instead of spawning ffmpeg per chunk.
6. Close encoder in `_finalize_job` with `persistent_encoder.finish()`.

Expected result:

- no more `h264_nvenc` broken pipe errors
- eliminate 144 ffmpeg process spawns per 8-stream test
- reduced encode latency per chunk
- lower CPU contention

## Phase 3: Scheduler and admission control

Goal:

- maximize safe throughput instead of blindly pushing concurrency

Files:

- `scripts/gpu_worker_service.py`
- `scripts/concurrent_gpu_manager.py`

Changes:

1. Add priority-aware queueing if needed.
2. Enforce hard cap on simultaneous live jobs.
3. Differentiate capacity by stream type if needed.
4. Reject or queue excess work explicitly.

Expected result:

- better p95 latency
- fewer cascading failures

## Phase 4: Encoder path optimization

Goal:

- reduce CPU chunk creation bottlenecks

Files:

- `scripts/api_avatar.py`
- possibly `scripts/hls_session_manager.py`

Changes:

1. Optional NVENC path for HLS segments.
2. Better ffmpeg cancellation and timeout reporting.
3. Segment encode metrics.

Expected result:

- lower CPU saturation
- faster chunk creation

## Phase 5: Horizontal scale design

Goal:

- support more users than one GPU can handle

Architecture:

1. one API layer
2. N GPU workers
3. central queue or router
4. session affinity if required

Changes:

1. route requests to a specific GPU worker
2. expose per-worker capacity and health
3. add load-aware placement

Expected result:

- multi-GPU and multi-host readiness

## File-By-File Change Plan

## `api_server.py`

Planned changes:

1. Add explicit request registration for all stream types.
2. Add cancel endpoints.
3. On delete, cancel active request before deleting session files.
4. Separate request submission from session deletion logic.
5. Expose request and worker stats endpoints.
6. If Phase 2 is adopted, replace direct executor calls with IPC to GPU worker service.

## `scripts/hls_session_manager.py`

Planned changes:

1. Extend `HlsSession` with cancellation and terminal metadata.
2. Add `mark_deleting()`.
3. Add `mark_cancel_requested()`.
4. Delay file deletion until worker exit or timeout.
5. Track live request linkage explicitly.

## `scripts/session_manager.py`

Planned changes:

1. Mirror the same request lifecycle fields for SSE sessions.
2. Separate session delete from request cancel.

## `scripts/webrtc_manager.py`

Planned changes:

1. Mirror the same request lifecycle fields for WebRTC sessions.
2. Ensure active live generation is cancelled before session teardown.

## `scripts/avatar_manager_parallel.py`

Planned changes:

1. Add per-avatar load locks. ✅ (completed)
2. Move request bookkeeping into a dedicated registry or service.
3. If staying monolith for a while, add proper done-callback cleanup.
4. In Phase 2, reduce this class to worker-side orchestration only.
5. Fix `compile_models()` to save dtypes before compilation and gate on `MUSETALK_COMPILE`.
6. Remove the duplicate `compile_models()` call from `_init_models()`.

## `scripts/api_avatar.py`

Planned changes:

1. Add `cancel_event`.
2. Add `progress_callback`.
3. Add request-specific `scratch_dir`.
4. Remove shared temp dir behavior from live streaming.
5. Ensure all cleanup happens in `finally`.
6. Report chunk encode and generation timings.
7. Pre-stack `input_latent_list_cycle` as a single tensor (`input_latent_cycle_tensor`) for fast GPU batch indexing in the scheduler.

## `scripts/hls_gpu_scheduler.py`

Completed changes:

1. Own shared HLS GPU scheduling.
2. Batch work across active HLS sessions up to a bounded combined batch size.
3. Separate prep work from GPU generation work.
4. Separate TS encode work from GPU generation work.
5. Report scheduler queue and job progress statistics.
6. `_memory_bucket` returns 1 to prevent deadlock.
7. Round 3 fill guard removed (always fills remaining GPU capacity).
8. Compose and encode workers auto-scale based on CPU count.
9. Batch padding for torch.compile compatibility (FIXED_SIZES = [4, 8, 16, 32]).

Planned changes:

1. Use `input_latent_cycle_tensor.index_select()` for fast batch assembly instead of per-frame `torch.cat`.
2. Bypass `gpu_memory.allocate()` entirely — the serial scheduler loop only needs one lease.
3. Integrate persistent per-stream ffmpeg encoder.

## `scripts/concurrent_gpu_manager.py`

Planned changes:

1. Track per-request leases.
2. Replace spin loop with wait/notify.
3. Expose waiting queue and holders.
4. Support queue timeout and admission control.

Current note:

- The HLS scheduler effectively bypasses this module by using `_memory_bucket=1`. Changes here primarily affect the SSE and non-HLS inference paths.

## `load_test.py`

Planned changes:

1. Add request-cancel coverage.
2. Add repeated-stage runs without server restart.
3. Capture stats before and after each stage.
4. Flag leaked active requests after a stage completes.
5. Report queue delay and cancellation latency.

## Recommended Metrics

At minimum, expose:

1. `active_requests`
2. `queued_requests`
3. `requests_by_state`
4. `gpu_budget_current_gb`
5. `gpu_budget_waiters`
6. `avg_time_to_first_chunk`
7. `avg_chunk_creation_time`
8. `request_cancellations`
9. `orphaned_request_count`
10. `worker_restart_count`
11. `avatar_cache_hits`
12. `avatar_cache_misses`
13. `avatar_load_wait_time`
14. `avg_batch_generation_time`
15. `avg_segment_encode_time`
16. `time_to_first_generated_frame`
17. `time_to_first_chunk`
18. `live_generation_slots_in_use`
19. `live_generation_queue_depth`
20. `request_last_progress_at`
21. `aggregate_gpu_fps` (total frames generated per second across all streams)
22. `per_stream_effective_fps` (aggregate_gpu_fps / active_streams)

## Recommended Status Codes And Semantics

Use explicit operational semantics:

1. `200`
   - request started immediately
2. `202`
   - cancellation accepted
3. `409`
   - session already streaming
4. `429`
   - queue full or capacity exceeded
5. `503`
   - worker unavailable or restarting

## Load Testing Acceptance Criteria

The system should not be considered fixed until all of the following pass:

1. Repeated `concurrency=4` runs can be executed back-to-back without restarting the backend.
2. After a timed-out or cancelled run, `active_requests` returns to zero.
3. After a cancelled HLS session, no stale request remains in `running` state.
4. GPU budget usage returns to baseline after all active work ends.
5. Session deletion does not leave chunk writers active.
6. Same-avatar concurrent cold start causes only one disk load.
7. No shared-temp-path collisions occur between concurrent requests.
8. `concurrency=1` sustains chunk cadence close to the target segment duration.
9. `concurrency=1` no longer reports throttling under the defined threshold.
10. `concurrency=2` can reach steady chunk production without first-chunk stall behavior.
11. Capacity claims are based on realtime chunk cadence, not only request completion.
12. `concurrency=8` with `max_combined_batch_size=32` completes all sessions without deadlock.
13. No `_memory_bucket` deadlock at any concurrency level.

## Throughput Scaling Acceptance Criteria

For 8-stream parity with single-stream speed:

1. Either `torch.compile` or TensorRT must be operational, providing at minimum 2.5x aggregate throughput improvement.
2. Or `musetalk_fps` must be lowered to a level where `8 × musetalk_fps ≤ aggregate_gpu_fps`.
3. Persistent per-stream encoding must replace per-chunk ffmpeg spawning.
4. No NVENC broken pipe errors during any test run.
5. `avg_segment_interval_s` at `concurrency=8` must be within 1.5x of `avg_segment_interval_s` at `concurrency=1`.

## Operational Guidance Until This Is Implemented

Until the above changes are in place:

1. Restart the backend after a failed high-concurrency load run.
2. Treat results after the first timeout as invalid.
3. Keep HLS concurrency conservative.
4. Warm the target avatar before testing.
5. Test repeated runs, not just one clean run.
6. Use the following launch configuration for best 8-stream behavior:

```bash
unset PYTORCH_CUDA_ALLOC_CONF
unset MUSETALK_COMPILE
export HLS_SCHEDULER_MAX_BATCH=32
export HLS_SCHEDULER_STARTUP_SLICE_SIZE=4
export HLS_SCHEDULER_AGGRESSIVE_FILL_MAX_ACTIVE_JOBS=999
export HLS_COMPOSE_WORKERS=8
export HLS_ENCODE_WORKERS=8
export HLS_MAX_PENDING_JOBS=24
python api_server.py --host 0.0.0.0 --port 8000
```

7. Do NOT set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the current PyTorch version does not support this option and will crash on startup.
8. Do NOT set `MUSETALK_COMPILE=1` until the `.dtype`/`.parameters()` access sites are fully audited and fixed.

## Recommended Order Of Execution

If you want the highest return on engineering time, do the work in this order:

1. cancellation and request registry ✅ (partially completed)
2. per-avatar load lock ✅ (completed)
3. request-specific scratch directories ✅ (completed)
4. request reaper and stats ✅ (partially completed)
5. `_memory_bucket=1` and scheduler fill guard removal ✅ (completed)
6. pre-stack avatar latents as tensor for fast batch assembly
7. persistent per-stream ffmpeg encoder
8. `torch.compile` integration (Phase 1.5)
9. bypass `gpu_memory.allocate()` in the scheduler
10. admission control
11. GPU worker process split
12. supervisor and restart logic
13. TensorRT integration (if torch.compile is insufficient)

## Final Recommendation

If the goal is "support as many users as possible", the best long-term architecture is:

1. API server handles HTTP and session state.
2. One dedicated GPU worker process per GPU handles model execution.
3. A supervisor monitors and restarts the GPU worker if it stalls.
4. Every request has explicit lifecycle, heartbeat, cancellation, and cleanup.
5. Admission control rejects or queues excess work instead of letting the system degrade unpredictably.

That design gives you:

- predictable throughput
- safe cancellation
- recoverability
- cleaner load testing
- a real path to multi-GPU scale

## March 15 Persistent Encoder And Player Findings

This section captures the latest experimental cycle after the startup-path refactor.

### What was tested

The HLS path was refactored to use a continuous per-request ffmpeg process instead of spawning ffmpeg once per segment.

Related changes and observations:

1. A persistent encoder path was introduced for HLS.
2. An NVENC session pool was added so overflow streams would start on `libx264` instead of failing hard.
3. Additional HLS wall and scheduler instrumentation was added to inspect first-chunk timing, buffer depth, encode backlog, and tail drain.

### What the experiment proved

The persistent encoder path did **not** materially improve steady-state throughput on the RTX 3090.

Representative `concurrency=8`, `playback_fps=24`, `musetalk_fps=12`, `batch_size=2` result after the manifest/path fixes:

```json
{
  "avg_time_to_live_ready_s": 8.733,
  "avg_segment_interval_s": 5.223,
  "max_segment_interval_s": 5.521,
  "wall_time_s": 96.3,
  "gpu": {
    "avg_util_pct": 33.05,
    "peak_util_pct": 100.0,
    "peak_memory_used_mb": 12221.0
  }
}
```

Interpretation:

1. This is effectively back in the same `~5.1s-5.3s` segment cadence band as before.
2. Therefore per-segment ffmpeg spawn overhead was not the main reason for the current `concurrency=8` throttle ceiling.
3. The deeper bottlenecks are still prep cost, scheduler turn cost, CPU compose, and the aggregate model-throughput ceiling.

### What broke during the experiment

The backend was still generating chunks, but the browser player stopped behaving reliably.

What actually happened:

1. The persistent ffmpeg path wrote a valid `live.m3u8`, but the playlist entries no longer matched the nested on-disk segment paths until the API serving layer was patched.
2. After that serving fix, the wall/player could still get stuck at `Preparing live...` because short streams completed before the player revealed live.
3. Additional player-side reveal gating changes then made the iframe state machine fragile enough that the wall became untrustworthy for evaluating backend throughput.

Important clarification:

- "chunks are not being generated" was a false signal.
- backend logs showed `first chunk ready` and final `chunks=4` / `chunks=5` style completion lines.
- the real issue was player / session-state handling after the recent HLS changes.

### NVENC pool conclusion

The NVENC session pool was still a useful correctness finding:

1. Persistent per-stream NVENC can exhaust encoder resources under `concurrency=8`.
2. Limiting long-lived NVENC sessions avoids the `OpenEncodeSessionEx failed` / `No capable devices found` failure mode.
3. However, falling overflow streams back to `libx264` can increase CPU pressure and does not solve the main throughput ceiling by itself.

### Metric interpretation warning

After the persistent encoder refactor, `avg_encode` in scheduler logs is no longer directly comparable to the older per-segment path.

Reason:

1. The old path measured per-segment encode task duration.
2. The persistent path measures frame submission / queue behavior into one long-lived ffmpeg process.

So the new `avg_encode` should not be used as proof that true encode cost collapsed.

### Current practical status

The current recommendation is:

1. Freeze or revert the recent player / live-HLS serving experiments to the last known-good UI state before doing more throughput work.
2. Keep throughput experiments backend-only whenever possible.
3. Treat the persistent encoder experiment as informative, but not yet production-ready.
4. Focus the next optimization pass on:
   - startup prep structure
   - CPU compose cost
   - model-path speed

### March 15 refactor result update

After the backend-only compose and PE refactor, the RTX 3090 produced a much better `concurrency=8`, `24/12`, `batch_size=2` run:

```json
{
  "avg_time_to_live_ready_s": 3.282,
  "avg_segment_interval_s": 2.301,
  "max_segment_interval_s": 3.521,
  "wall_time_s": 44.4,
  "gpu": {
    "avg_util_pct": 75.18,
    "peak_util_pct": 100.0,
    "avg_memory_used_mb": 8998.2,
    "peak_memory_used_mb": 10588.0
  }
}
```

Relative to the prior reverted-baseline result around:

- `avg_time_to_live_ready_s ≈ 8.733`
- `avg_segment_interval_s ≈ 5.223`
- `wall_time_s ≈ 96.3`

this is a major step forward:

1. startup became roughly 2.7x faster
2. steady-state cadence became roughly 2.3x better
3. wall time became roughly 2.2x better
4. average GPU utilization rose sharply, which indicates the GPU is being fed much more consistently

Representative server-side scheduler timings for the same run:

- `avg_gpu_batch ≈ 0.727s to 0.813s`
- `avg_compose ≈ 0.122s to 0.140s`
- `avg_encode ≈ 0.466s to 0.681s`
- `first_chunk ≈ 2.04s to 3.67s`

Interpretation:

1. The old compose implementation was a real bottleneck.
2. Repeated PE work inside the live GPU loop was also a real bottleneck.
3. The current system is still technically throttled at `concurrency=8`, but it is now in a much healthier regime.
4. The next bottleneck investigation should be based on this new baseline, not the older `~5.2s` interval runs.

This is not square one. The experiment ruled out one major hypothesis:

- per-segment ffmpeg spawning is **not** the primary reason `concurrency=8` remains throttled on the RTX 3090.

## March 15 Post-Revert Throughput Plan

After reverting the player and serving experiments, the HLS browser path is back to generating and exposing live chunks correctly. The remaining problem is backend cadence: under concurrent load, the system still produces live media too slowly to stay ahead of playback, so the browser buffer drains.

Operational rule:

- if `avg_segment_interval_s > segment_duration`, buffering is inevitable
- player tuning can smooth presentation, but it cannot remove a backend cadence deficit

### Ranked bottlenecks after revert

#### 1. CPU compose cost is the highest-confidence software bottleneck

The compose hot path in `scripts/api_avatar.py` and `musetalk/utils/blending.py` still performs expensive per-frame work:

- full frame copy
- `cv2.resize(...)` on each frame
- PIL conversion of the base frame and talking ROI
- masked paste / blend
- conversion back to NumPy

Recent scheduler timing has shown `avg_compose` in the same general range as `avg_gpu_batch`, while compose queue wait remains low. That means the work itself is expensive, not merely underprovisioned.

Current recommendation:

1. Rewrite compose to stay NumPy / OpenCV only.
2. Operate only on the ROI instead of rebuilding full PIL images.
3. Reuse precomputed mask arrays and crop coordinates directly.
4. Treat more compose workers only as a temporary queueing tool, not as the real fix.

#### 2. Whisper prep still competes with live generation on the same GPU

`scripts/hls_gpu_scheduler.py` still performs full audio feature extraction and full Whisper prompt generation during `_prepare_job(...)`.

`musetalk/utils/audio_processor.py` still runs `whisper.encoder(...)` on the main generation GPU.

That means concurrent stream starts can steal GPU time from streams that are already live.

Current recommendation:

1. Prepare only enough audio / Whisper prompts for the first chunk or two.
2. Continue the rest of prep in the background after the stream is admitted.
3. If startup interference remains severe, move Whisper prep to CPU or a separate GPU path.

#### 3. PE is still paid every scheduler turn

The shared HLS scheduler still runs `self.manager.pe(audio_inputs)` inside every generation batch.

Once Whisper prompts are known, PE output is deterministic and can be precomputed.

Current recommendation:

1. Precompute PE outputs during prep or just-in-time chunk staging.
2. Feed the scheduler precomputed PE tensors instead of raw Whisper prompts.
3. Re-measure `avg_gpu_batch` after PE is removed from the live loop.

#### 4. Batch assembly still has Python and copy overhead

The scheduler still builds latent index lists in Python, concatenates tensors on CPU, and copies assembled batches to GPU every turn.

Current recommendation:

1. Reduce Python-side latent index construction.
2. Use more contiguous tensor gathering for latent cycles.
3. Pin or stage frequently reused CPU tensors where practical.
4. Measure whether assembly and copy time fall after these changes.

#### 5. VAE decode forces an immediate GPU-to-CPU sync

`musetalk/models/vae.py` still converts decoded batches immediately to CPU NumPy arrays.

That forces synchronization and locks the current architecture into CPU compose.

Current recommendation:

1. Keep this as a later, larger refactor.
2. If compose optimization alone is not enough, evaluate keeping decoded tensors on GPU longer and moving more of compose off CPU.

#### 6. Per-chunk ffmpeg remains overhead, but it is not the first thing to chase

The reverted baseline still spawns ffmpeg per segment. That cost is real, but the earlier persistent-encoder experiment showed it was not the primary reason for the `~5.2s` steady-state segment cadence on the RTX 3090.

Current recommendation:

1. Do not make player / serving changes the primary focus right now.
2. Revisit encode architecture only after compose and prep competition are addressed.

### Recommended implementation order

1. Optimize CPU compose in `scripts/api_avatar.py` and `musetalk/utils/blending.py`.
2. Stop full upfront GPU Whisper prep from competing with live generation.
3. Precompute PE outputs so the live scheduler pays less per turn.
4. Tighten scheduler batch assembly and copy overhead.
5. Re-test `torch.compile` or a newer PyTorch / CUDA stack only after the above are measured.
6. Consider GPU-side compose / decode retention only if the simpler wins are not enough.

### What to avoid in the next cycle

1. Do not reopen HLS player or manifest experiments while backend throughput is still the limiting factor.
2. Do not assume more compose or encode workers will materially change the ceiling when queue wait is already low.
3. Do not expect `batch_size` tuning alone to remove buffering at `concurrency=8`.
4. Do not treat VRAM headroom as evidence that the machine has more realtime throughput available.

### March 15 priority status update

The original post-revert priorities have now been partially executed. The current scoreboard is:

| Priority | Status | Current read |
| --- | --- | --- |
| 1. CPU compose optimization | ✅ Major pass complete | This was the biggest measured win. `avg_compose` dropped from roughly `~1.1s` to about `~0.145s`, and the `concurrency=8` cadence improved from the older `~5.2s` band to the newer `~2.3-2.4s` band. |
| 2. Stop Whisper prep from stealing GPU time | 🟡 Partially complete | Conditioning prep is now incremental and backfilled in the background, which improved startup behavior and reduced up-front prep pressure. Whisper encode still uses the main GPU, so this priority is improved but not fully solved. |
| 3. Precompute PE | ✅ Complete for current architecture | PE was removed from the live GPU loop and moved into CPU-side prep. This was a real contributor to the large March 15 throughput gain. |
| 4. Reduce scheduler assembly / copy overhead | 🟡 Partial, low measured impact so far | Latent tensorization, pinned tensors, and staging buffers are now in place. The most recent measurements did not show a clear additional reduction in `avg_gpu_batch`, so this path no longer looks like the next step-change. |
| 5. Revisit model acceleration | 🟡 Partially complete | `torch.compile` integration is now working in a safer per-module form, and compiled model-path throughput materially improved the warm-cache `concurrency=8` result. This is no longer speculative, but it is not fully solved yet because the system is still slightly above the realtime threshold and compile remains stack-sensitive. |
| 6. Keep decode / composition off CPU longer | ⏳ Not complete | `musetalk/models/vae.py` still converts decoded output straight to CPU NumPy. This remains a larger architectural refactor to consider only after model acceleration is measured. |

Current interpretation after the latest warm `concurrency=8`, `24/12`, `batch_size=2` runs:

- `avg_segment_interval_s` is now stable in the `~2.35-2.40s` range
- `avg_time_to_live_ready_s` is now stable in the `~3.1-3.4s` range
- `wall_time_s` is now stable around `~44.2-44.5s`
- the system is still throttled, but it is no longer in the old overloaded `~5.2s` cadence regime

Most important current conclusion:

- the backend is no longer primarily limited by CPU compose
- the remaining bottleneck is mostly **GPU turn cadence / model-path throughput**
- therefore the next engineering priority should be **Priority 5**, not more player work and not more compose tuning

### March 17 compile integration update

A dedicated `torch.compile` refactor was completed in `scripts/avatar_manager_parallel.py` and `musetalk/models/vae.py` with three important changes:

1. **Per-module compile fallback** instead of all-or-nothing restore. A bad VAE compile no longer forces the UNet back to eager.
2. **Safer compile defaults** (`reduce-overhead` first) plus explicit warmup and traceback logging.
3. **VAE tensor decode warmup path** so compile warmup no longer depends on the CPU/NumPy conversion path.

An additional March 17 bug was found and fixed during validation:

1. the initial UNet warmup used a dummy latent tensor with **4 channels**
2. the MuseTalk UNet actually expects **8 input channels**
3. this caused the first compile attempt to fail during warmup at `conv_in`
4. after switching the warmup tensor to use the model's real `in_channels`, the UNet warmup no longer failed immediately

Measured impact on the RTX 3090 after the compile integration:

- previous warm-cache reference point:
  - `avg_segment_interval_s ≈ 2.35-2.40`
  - `avg_time_to_live_ready_s ≈ 3.1-3.4`
  - `wall_time_s ≈ 44.2-44.5`
  - `avg_gpu_batch ≈ 0.788-0.791s`
- March 17 compile-enabled runs:
  - run 1:
    - `avg_time_to_live_ready_s = 3.670`
    - `avg_segment_interval_s = 2.062`
    - `max_segment_interval_s = 3.562`
    - `wall_time_s = 39.1`
    - `avg_gpu_util = 86.72%`
  - run 2:
    - `avg_time_to_live_ready_s = 3.455`
    - `avg_segment_interval_s = 2.076`
    - `max_segment_interval_s = 3.106`
    - `wall_time_s = 39.2`
    - `avg_gpu_util = 86.7%`
- server-side scheduler logs during the compile-enabled run:
  - `avg_gpu_batch ≈ 0.691s`
  - `avg_compose ≈ 0.154-0.158s`
  - `avg_encode ≈ 0.770-0.806s`

Current March 17 interpretation:

1. compile **did** improve throughput; it was not a no-op
2. `avg_gpu_batch` improved by about `12%` (`~0.79s -> ~0.69s`)
3. `avg_segment_interval_s` improved by about `13%` (`~2.38s -> ~2.07s`)
4. `wall_time_s` improved by about `11-12%` (`~44.3s -> ~39.1s`)
5. the system is still slightly throttled because the current pipeline still needs about **3 scheduler turns per segment**, and the tail still includes meaningful encode cost and jitter

This changes the practical status of Priority 5:

- model-path acceleration is now **partially validated**
- the remaining gap is no longer "make compile work at all"
- the remaining gap is "reduce the next exposed bottlenecks after compile", especially scheduler-turn math and encode/tail jitter

### March 17 scheduler policy update

After the compile-enabled baseline was established, the HLS scheduler allocation policy in `scripts/hls_gpu_scheduler.py` was refactored again so spare GPU capacity is used to finish jobs that are closest to emitting their next HLS chunk, instead of only spreading small equal slices across all warmed streams.

Measured impact on the RTX 3090, using warm-cache `concurrency=8`, `playback_fps=24`, `musetalk_fps=12`, `batch_size=2`:

- previous compile-enabled reference point:
  - `avg_time_to_live_ready_s ≈ 3.46-3.67`
  - `avg_segment_interval_s ≈ 2.06-2.08`
  - `wall_time_s ≈ 39.1-39.2`
  - `avg_gpu_batch ≈ 0.691s`
- post scheduler-policy runs:
  - run 1:
    - `avg_time_to_live_ready_s = 3.598`
    - `avg_segment_interval_s = 1.944`
    - `max_segment_interval_s = 3.030`
    - `wall_time_s = 38.5`
    - `avg_gpu_util = 82.53%`
  - run 2:
    - `avg_time_to_live_ready_s = 3.458`
    - `avg_segment_interval_s = 1.984`
    - `max_segment_interval_s = 3.039`
    - `wall_time_s = 38.2`
    - `avg_gpu_util = 84.75%`
- representative server-side scheduler timings after the scheduler-policy change:
  - `avg_gpu_batch ≈ 0.675-0.676s`
  - `avg_compose ≈ 0.085-0.097s`
  - `avg_encode ≈ 0.583-0.609s`

Interpretation:

1. The scheduler policy refactor produced another real win on top of `torch.compile`.
2. Average segment cadence is now at or just under the strict 1-second-segment realtime target for `concurrency=8`.
3. The remaining warning is now mostly about **worst-case spikes**, not the average path.
4. The next exposed bottlenecks are now:
   - late-stream startup / prep skew
   - chunk encode and tail jitter

### March 18 shared-batch retuning update

The next backend-only experiment on March 18 was to make larger combined scheduler batch sizes first-class shapes instead of leaving everything above `32` in the "allowed but not specifically compile-warmed" path.

Changes:

1. `scripts/hls_gpu_scheduler.py` no longer stops its fixed padded batch sizes at `[4, 8, 16, 32]`.
2. The scheduler now extends those fixed shapes from `HLS_SCHEDULER_MAX_BATCH`, so `48` becomes a real compile-friendly target instead of an ad-hoc actual-batch shape.
3. `scripts/avatar_manager_parallel.py` now mirrors that logic in the default `MUSETALK_COMPILE_WARMUP_BATCHES` path, so the compile warmup and the scheduler agree on the same larger shapes.

Measured effect on the RTX 3090 using warm-cache `concurrency=8`, `playback_fps=24`, `musetalk_fps=12`:

- previous `HLS_SCHEDULER_MAX_BATCH=32`, `batch_size=4` reference:
  - `avg_time_to_live_ready_s = 3.652`
  - `avg_segment_interval_s = 2.034`
  - `max_segment_interval_s = 3.183`
  - `wall_time_s = 38.9`
- March 18 `HLS_SCHEDULER_MAX_BATCH=48`, `batch_size=4`:
  - `avg_time_to_live_ready_s = 4.648`
  - `avg_segment_interval_s = 1.965`
  - `max_segment_interval_s = 3.129`
  - `wall_time_s = 39.1`
- best observed March 18 near-pass run with the same `HLS_SCHEDULER_MAX_BATCH=48`, `batch_size=4` settings:
  - `avg_time_to_live_ready_s = 4.897`
  - `avg_segment_interval_s = 1.921`
  - `max_segment_interval_s = 2.046`
  - `wall_time_s = 38.5`
- previous `HLS_SCHEDULER_MAX_BATCH=32`, `batch_size=8` reference:
  - `avg_time_to_live_ready_s = 3.682`
  - `avg_segment_interval_s = 1.891`
  - `max_segment_interval_s = 4.557`
  - `wall_time_s = 37.6`
- March 18 `HLS_SCHEDULER_MAX_BATCH=48`, `batch_size=8`:
  - `avg_time_to_live_ready_s = 4.650`
  - `avg_segment_interval_s = 1.881`
  - `max_segment_interval_s = 4.095`
  - `wall_time_s = 37.5`
- GPU memory impact at `HLS_SCHEDULER_MAX_BATCH=48`:
  - `peak_memory_used_mb ≈ 22.3-22.5 GB`

Interpretation:

1. Raising the **total combined scheduler batch** from `32` to `48` produced another real steady-state throughput gain.
2. The gain is clearest in the `batch_size=4` comparison: `avg_segment_interval_s` improved from `2.034` to `1.965`, which puts the 8-stream average essentially at the realtime boundary for 1-second segments.
3. The larger combined batch does **not** automatically improve startup. `avg_time_to_live_ready_s` became worse because each scheduler turn is now heavier and later jobs can wait longer before their first productive turn.
4. Increasing per-stream `batch_size` is still a fairness tradeoff, not a pure speed knob. Higher request `batch_size` can improve the average by letting some streams grab larger slices, while still making late or unlucky streams worse.
5. `HLS_SCHEDULER_MAX_BATCH=48` now looks viable on the RTX 3090, but it pushes VRAM very close to the wall. `64` should be treated as risky until proven otherwise on this exact stack.
6. The best observed `batch_size=4` run was effectively a near-pass. `max_segment_interval_s = 2.046` is only `46ms` over the warning threshold, which means the system is now capable of getting very close to clean 8-stream realtime behavior on average.
7. Identical `HLS_SCHEDULER_MAX_BATCH=48`, `batch_size=4` runs can still vary in `max_segment_interval_s` because the extra 16-frame chunk-completion budget gets distributed differently depending on tiny startup and queue-order differences. The average path stays similar, but one unlucky stream can still miss a favorable turn and become the worst-case outlier.
8. A strong next experiment is inferred from these results: keep `HLS_SCHEDULER_MAX_BATCH=48`, but return request `batch_size` to `2` to see whether the larger total batch can be kept while restoring better fairness and startup spread.

### Current next priorities

With the March 18 shared-batch retuning results in place, the latest backend priority order is:

1. **Retune shared batch throughput** around the compiled path (`HLS_SCHEDULER_MAX_BATCH`, fixed scheduler shapes, and request `batch_size` fairness)
2. **Reduce late-stream startup skew** in `scripts/hls_gpu_scheduler.py` and `musetalk/utils/audio_processor.py`
3. **Reduce encode / tail jitter** in `scripts/api_avatar.py`
4. **Continue model-path acceleration** in `scripts/avatar_manager_parallel.py` only if the above still leaves a gap
5. **Consider larger GPU-native architecture work** in `musetalk/models/vae.py` only after the lower-risk backend steps are exhausted

### Short operational read

At the moment, the safest mental model is:

- the browser buffers because the backend is producing live media too slowly
- the biggest remaining software opportunity is now **shared scheduler batch retuning plus startup-skew cleanup**
- player tuning can improve perceived smoothness, but not remove a backend cadence deficit

## Current Bottom Line

Based on the full debugging cycle so far:

1. The original orphan-worker problem was real and required lifecycle fixes.
2. Those fixes were necessary, but they did not solve realtime throughput.
3. The HLS path no longer runs one GPU-driving thread per stream; it now uses a shared scheduler plus separate encode workers.
4. That new scheduler materially improved startup behavior and completion reliability.
5. The backend is still slower than realtime even for a single HLS stream under the current tuned settings.
6. `concurrency=2` can now complete, but not at healthy realtime chunk cadence.
7. `concurrency=3` reveals fairness problems and unacceptable tail latency.
8. Therefore the next engineering priority is profiling and tuning the shared scheduler path, not blind increases in concurrency.
9. The system should not claim higher user counts until the shared HLS scheduler can reliably meet the realtime target.
10. The `_memory_bucket` deadlock was fixed by returning 1, enabling 8-stream completion.
11. The earlier `~18-20 fps` aggregate ceiling estimate is now stale; the March 17 compile-enabled runs show the current tuned stack can materially exceed the older post-revert baseline.
12. Achieving stable 8-stream realtime still requires either better shared-batch utilization, less startup skew, fewer turns per segment, or less encode/tail jitter.
13. The persistent encoder experiment improved understanding of the pipeline, but did not materially improve steady-state throughput.
14. The March 18 `HLS_SCHEDULER_MAX_BATCH=48` retune materially improved the 8-stream average again, bringing `concurrency=8` much closer to effective realtime on average even though startup spread remains worse.
15. The most practical near-term path to 8-stream parity now appears to be: keep the compiled path, keep larger combined scheduler batches, and recover fairness/stability by tuning request `batch_size` and startup behavior rather than assuming bigger per-stream slices are universally better.
