# MuseTalk Autoscaling Plan

This document is the MuseTalk-specific autoscaling plan for a React Native
callAnnie-like application.

It assumes:

- React Native talks only to the main EC2 backend
- EC2 is the control plane
- MuseTalk workers run on Vast.ai
- Chatterbox is a separate service with its own scaling policy

The goal here is not cross-service prediction. The goal is to scale MuseTalk
based on MuseTalk demand and MuseTalk capacity.

## Decision Summary

MuseTalk and Chatterbox should be treated as separate services.

That means:

- Chatterbox throughput should not be the main scaling signal for MuseTalk
- MuseTalk should scale from its own queue depth, active session count,
  in-flight generations, and worker health
- MuseTalk workers should send their own capacity and status data back to EC2
- EC2 should decide when to boot more Vast.ai workers

Recommended control path:

```text
React Native App
    |
    v
AWS EC2 Backend API
    |
    +--> job queue
    +--> worker registry
    +--> session router
    +--> autoscaler
    |
    +--> MuseTalk Vast.ai pool
    |
    +--> S3
```

## Why MuseTalk Needs Its Own Autoscaling Logic

MuseTalk is the more stateful service in your stack.

In the current server design, a MuseTalk node may hold:

- in-memory sessions
- active SSE chunk streams
- HLS session state and local segment files
- WebRTC peer connections and live tracks
- GPU scheduler state

That means EC2 cannot scale MuseTalk based only on generic traffic numbers.
It needs MuseTalk-specific signals from the workers themselves.

## Current Repo Signals We Can Reuse

The current MuseTalk server already exposes several useful endpoints and
internal concepts that map well to autoscaling.

Useful existing endpoints:

- `/health`
- `/stats`
- `/stats/gpu-live`
- `/sessions/live`
- `/hls/sessions/stats`

Useful existing capacity concepts:

- active live sessions
- active requests
- compute slots
- HLS scheduler queue depth
- GPU memory usage
- preparing jobs vs queued jobs

That means we do not need to invent the whole telemetry model from scratch.
We mainly need to standardize what gets sent to EC2.

## High-Level Architecture

### Public path

The mobile app should call only EC2.

Examples:

- `POST /avatar/jobs`
- `POST /musetalk/sessions`
- `POST /musetalk/sessions/:id/stream`
- `GET /avatar/jobs/:jobId`
- `GET /avatar/jobs/:jobId/result`

### Internal path

EC2 should:

- create jobs
- assign a worker for async render jobs
- assign and pin a worker for live MuseTalk sessions
- store `session_id -> worker_id`
- poll or receive worker heartbeats
- run the autoscaler loop
- create or terminate Vast instances

### Worker path

Each MuseTalk worker should:

- boot on a fixed internal app port such as `8000`
- discover its mapped Vast public port
- register with EC2 once healthy
- send regular heartbeats
- report capacity and session stats
- support drain mode

## Recommended Worker States

Use explicit worker lifecycle states:

- `booting`
- `registering`
- `ready`
- `busy`
- `draining`
- `unhealthy`
- `terminated`

These states should live in the EC2 worker registry, not just inside the
worker process.

## Registration Contract

Each MuseTalk worker should register itself with EC2 after it is healthy.

Example:

```json
{
  "worker_id": "musetalk-33678878",
  "worker_type": "musetalk",
  "vast_instance_id": "33678878",
  "endpoint_url": "http://154.61.62.158:50401",
  "public_ip": "154.61.62.158",
  "public_port": 50401,
  "internal_port": 8000,
  "gpu_type": "RTX 3090",
  "status": "healthy",
  "max_concurrency": 1,
  "version": "2026-03-30.1"
}
```

Important:

- `endpoint_url` must use the Vast-mapped external port
- EC2 should never assume that public port equals internal `8000`

## Heartbeat Contract

The worker heartbeat should be the main way EC2 learns whether the cluster is
approaching saturation.

Suggested heartbeat interval:

- every `15s` during normal operation
- every `5s` when the worker is `busy` or `draining`

Recommended heartbeat payload:

```json
{
  "worker_id": "musetalk-33678878",
  "status": "healthy",
  "draining": false,
  "active_jobs": 1,
  "active_live_sessions": 1,
  "active_sse_streams": 0,
  "active_hls_streams": 1,
  "active_webrtc_streams": 0,
  "max_concurrency": 1,
  "free_capacity": 0,
  "queue_depth": 2,
  "gpu_util_pct": 91.0,
  "memory_used_mb": 21500,
  "memory_total_mb": 24564,
  "avg_job_ms": 14200,
  "p95_first_chunk_ms": 2800,
  "last_error": null
}
```

## What Data Should Be Sent To EC2

MuseTalk workers should send data that answers four questions:

1. Is the worker healthy?
2. Can it accept another job right now?
3. Is it already falling behind?
4. Is it safe to terminate?

Recommended source-of-truth fields:

- `status`
- `draining`
- `active_jobs`
- `active_live_sessions`
- `free_capacity`
- `queue_depth`
- `gpu_util_pct`
- `memory_used_mb`
- `avg_job_ms`
- `p95_first_chunk_ms`
- `last_heartbeat_at`

Recommended advanced fields:

- `active_sse_streams`
- `active_hls_streams`
- `active_webrtc_streams`
- `preparing_jobs`
- `hls_scheduler_pending_jobs`
- `compute_slots_in_use`
- `compute_slots_free`
- `session_count`
- `oldest_queued_job_ms`

## How The Worker Gets This Data

There are two good patterns.

### Option A: Local sidecar or agent on the worker

A small worker-side agent:

- calls the local MuseTalk endpoints
- builds a normalized heartbeat payload
- sends it to EC2

This is the cleanest option because EC2 does not have to know the details of
every MuseTalk endpoint shape.

### Option B: EC2 polls the worker directly

EC2 can call:

- `/health`
- `/stats`
- `/stats/gpu-live`
- `/sessions/live`
- `/hls/sessions/stats`

This works, but it couples EC2 more tightly to the worker API.

### Recommended approach

Use both:

- worker push heartbeat as the main control signal
- EC2 verification polling as a secondary trust check

That gives you better freshness without fully trusting unverified worker state.

## Session Routing Requirements

MuseTalk live flows must use sticky routing.

When EC2 creates a session, it must:

1. choose a worker
2. create the session on that worker
3. store `session_id -> worker_id`
4. route all future calls for that session to the same worker

This applies especially to:

- SSE session streaming
- HLS sessions
- WebRTC sessions

Without session pinning, the current MuseTalk architecture will break because
session state is held on the worker.

## Async Jobs Vs Live Sessions

You should treat these as separate scheduling classes.

### Async render jobs

Examples:

- generate a final talking avatar video
- no persistent live session
- can be queued and retried more easily

Scheduling goal:

- maximize throughput
- queue when needed
- retry safely on another worker if a worker dies before output is committed

### Live session jobs

Examples:

- active avatar conversation
- user waiting for near-realtime response
- HLS or WebRTC session already created

Scheduling goal:

- low queue time
- strong sticky routing
- conservative concurrency

This distinction matters because live session pressure should generally trigger
faster scale-up than async job pressure.

## Core Scale-Up Signals

EC2 should scale MuseTalk up when any of the following is true:

- queued MuseTalk jobs > threshold
- no ready worker exists
- no worker has `free_capacity > 0`
- active live sessions are close to total live capacity
- p95 queue wait exceeds target
- HLS scheduler pending jobs remain elevated
- repeated `429` responses indicate capacity exhaustion

Suggested first-pass rules:

- if `ready_workers == 0`, add `1` worker immediately
- if `queued_jobs >= 1` and `total_free_capacity == 0` for `30s`, add `1`
- if `queued_jobs >= 3`, add `2`
- if `live_capacity_utilization >= 0.85` for `30s`, add `1`
- if `p95_queue_wait_ms > 5000`, add `1`

## Core Scale-Down Signals

EC2 should scale MuseTalk down only when all are true:

- worker is `healthy` or `draining`
- worker has `active_jobs = 0`
- worker has `active_live_sessions = 0`
- worker has been idle for `10m to 20m`
- total healthy workers remain above minimum pool size

Suggested first-pass rules:

- minimum workers: `1`
- warm idle target: `1`
- maximum workers: `5`
- scale-up cooldown: `2m`
- scale-down cooldown: `10m`
- idle timeout before drain: `15m`

## Drain And Termination Flow

Never terminate a MuseTalk worker immediately if it may still hold live state.

Recommended drain flow:

1. EC2 marks worker `draining`
2. scheduler stops assigning new jobs to it
3. scheduler stops assigning new live sessions to it
4. worker finishes existing jobs and sessions
5. worker reports `active_jobs = 0` and `active_live_sessions = 0`
6. EC2 deregisters worker
7. EC2 terminates the Vast instance

This matters more for MuseTalk than for simpler stateless services.

## Recommended Capacity Formula

Use a MuseTalk-only desired worker formula.

Start simple:

```text
desired_workers =
  max(
    min_workers,
    min(
      max_workers,
      ceil(queued_jobs / jobs_per_worker_target) + warm_idle_target
    )
  )
```

Then improve it for live capacity:

```text
desired_workers =
  max(
    workers_for_async_queue,
    workers_for_live_sessions,
    min_workers
  )
```

Where:

```text
workers_for_live_sessions =
  ceil(active_live_sessions / live_sessions_per_worker_target) + warm_idle_target
```

Recommended starting assumptions:

- `jobs_per_worker_target = 1`
- `live_sessions_per_worker_target = 1`

That is intentionally conservative for the first rollout.

## How EC2 Should Trigger New Vast Workers

The worker should not directly decide to boot more workers.

Instead:

1. MuseTalk worker sends heartbeat to EC2
2. EC2 stores worker state in Redis or Postgres
3. autoscaler loop runs every `15s` or `30s`
4. autoscaler computes `desired_workers`
5. if `desired_workers > current_workers`, EC2 calls the Vast API to create more instances

That is the correct place for the "alert" behavior.

Optional fast path:

- if a worker reports `free_capacity = 0` and `queue_depth > 0`, EC2 can mark a
  temporary `capacity_hot` flag immediately
- the next autoscaler tick uses that flag to scale faster

But the final authority should still live on EC2.

## Suggested Database Tables

### workers

Store:

- worker identity
- endpoint
- status
- draining flag
- capacity
- heartbeat freshness

### worker_metrics

Store snapshots for:

- active jobs
- active live sessions
- queue depth
- GPU utilization
- memory utilization
- avg job time

### musetalk_sessions

Store:

- `session_id`
- `worker_id`
- `session_type`
- `user_id`
- `status`

This is the key table or cache for sticky routing.

### jobs

Store:

- job ID
- job type
- assigned worker
- status
- result URL
- failure reason

## Suggested MuseTalk Worker API Additions

For control-plane operations, it would help to add:

- `GET /ready`
- `POST /internal/drain`
- `GET /internal/capacity`

`/internal/capacity` could return a compact snapshot:

```json
{
  "status": "healthy",
  "draining": false,
  "active_jobs": 1,
  "active_live_sessions": 1,
  "free_capacity": 0,
  "queue_depth": 2,
  "compute_slots_in_use": 1,
  "compute_slots_free": 0
}
```

This can either be sent by the worker agent or polled by EC2.

## Recommended Rollout Order

### Phase 1

- add EC2 worker registry for MuseTalk
- add MuseTalk worker registration
- add MuseTalk worker heartbeat
- add session pinning in EC2
- add manual scale-up and drain support

### Phase 2

- add autoscaler loop
- add queue-based scale-up
- add idle-based scale-down
- add metrics dashboard

### Phase 3

- split async vs live scheduling policy
- add separate live-capacity alerts
- add faster hot-path scaling for session pressure

## Final Recommendation

Your revised instinct is right:

- Chatterbox and MuseTalk should scale independently
- MuseTalk needs its own autoscaling plan
- MuseTalk workers should send capacity and session data to EC2
- EC2 should be the thing that decides when to boot more Vast.ai workers

That gives you one stable control plane, one stable mobile contract, and an
autoscaler that is aligned with the actual behavior of the MuseTalk service.
