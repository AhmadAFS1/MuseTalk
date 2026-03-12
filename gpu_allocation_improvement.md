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

#### Updated performance conclusion

The new HLS scheduler clearly improved:

1. lifecycle safety
2. startup consistency
3. ability to complete concurrent jobs without obvious worker corruption

But the latest measurements now show a more nuanced picture:

1. single-stream cadence is healthy under the tested 1-second segment profile
2. two-stream cadence is close to realtime, but still near the throttle threshold
3. three-stream behavior is much better than before, but still too slow for a clean realtime claim

So the system has moved from "broken under concurrency" to "healthy at one stream, close at two, and still throughput-limited at three."

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

Relevant file:

- `scripts/concurrent_gpu_manager.py`

### 4. Same-avatar load race

`_get_or_load_avatar()` checks the cache, and if there is a miss, multiple concurrent requests can all load the same avatar independently.

Impact:

- Duplicate disk I/O.
- Slow cold-start behavior.
- More memory pressure.
- More variance during load tests.

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

Expected result:

- better visibility
- deterministic behavior under cancellation
- clear capacity limits

Important note:

- based on current measurements, memory admission and throughput admission must remain separate controls
- the new HLS scheduler is the correct architectural direction, but it still needs measurement and tuning before claiming safe two-stream realtime capacity

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

1. Add per-avatar load locks.
2. Move request bookkeeping into a dedicated registry or service.
3. If staying monolith for a while, add proper done-callback cleanup.
4. In Phase 2, reduce this class to worker-side orchestration only.

## `scripts/api_avatar.py`

Planned changes:

1. Add `cancel_event`.
2. Add `progress_callback`.
3. Add request-specific `scratch_dir`.
4. Remove shared temp dir behavior from live streaming.
5. Ensure all cleanup happens in `finally`.
6. Report chunk encode and generation timings.

## `scripts/hls_gpu_scheduler.py`

1. Own shared HLS GPU scheduling.
2. Batch work across active HLS sessions up to a bounded combined batch size.
3. Separate prep work from GPU generation work.
4. Separate TS encode work from GPU generation work.
5. Report scheduler queue and job progress statistics.

## `scripts/concurrent_gpu_manager.py`

Planned changes:

1. Track per-request leases.
2. Replace spin loop with wait/notify.
3. Expose waiting queue and holders.
4. Support queue timeout and admission control.

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

## Operational Guidance Until This Is Implemented

Until the above changes are in place:

1. Restart the backend after a failed high-concurrency load run.
2. Treat results after the first timeout as invalid.
3. Keep HLS concurrency conservative.
4. Warm the target avatar before testing.
5. Test repeated runs, not just one clean run.

## Recommended Order Of Execution

If you want the highest return on engineering time, do the work in this order:

1. cancellation and request registry
2. per-avatar load lock
3. request-specific scratch directories
4. request reaper and stats
5. admission control
6. GPU worker process split
7. supervisor and restart logic
8. encoder optimization

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
