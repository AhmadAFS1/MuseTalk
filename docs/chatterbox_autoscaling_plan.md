# Chatterbox-Fastest Autoscaling Plan

This document explains how a `chatterbox-fastest` worker pool should fit into
the AWS EC2 control plane and how it can make the overall React Native avatar
system easier to scale.

It is intentionally written as a control-plane document, not a model-quality
document. The main goal is predictable operations.

## Decision Summary

Use `chatterbox-fastest` as:

- the first autoscaled worker pool to productionize
- the simpler stateless pool that validates worker registration, heartbeat,
  draining, and queue-based scaling
- a separate service with its own queue and its own scaling policy
- the TTS pool that can be operated independently from MuseTalk

Recommended public flow:

```text
React Native App
    |
    v
AWS EC2 Backend API
    |
    +--> Job queue + worker registry + autoscaler
    |
    +--> Chatterbox-fastest pool
    |
    +--> MuseTalk pool
    |
    +--> S3 for audio/video assets
```

The mobile app should never know which Chatterbox worker or MuseTalk worker
handled a request. It should only know your stable backend API domain.

## Why Chatterbox Helps The Autoscaling Story

Chatterbox is easier to scale than MuseTalk because it is usually:

- more stateless
- less session-oriented
- faster to complete per job
- easier to retry on a different worker
- less sensitive to sticky routing than HLS or WebRTC

That makes it the best place to prove the autoscaling control plane first.

If the EC2 backend can successfully do the following for Chatterbox, the same
control-plane patterns can be reused for MuseTalk:

- worker self-registration
- worker heartbeats
- health and readiness checks
- queue-based scheduling
- scale-up and scale-down
- drain-before-terminate
- retry on alternate worker

In other words, Chatterbox is the low-risk place to build the autoscaling
machinery that MuseTalk will later depend on.

## How Chatterbox Fits In The System

Chatterbox should be treated as a separate service from MuseTalk.

That means:

- it should have its own worker pool
- it should have its own autoscaling policy
- it should not be the source-of-truth scaling signal for MuseTalk

It still helps the overall architecture by keeping the media pipeline staged
and observable.

### 1. Chatterbox keeps the pipeline staged

`chatterbox-fastest` should return `audio_duration_ms` with every successful
result.

The backend can separate the pipeline into stages:

1. text generation
2. TTS generation
3. avatar render

That creates a natural queue boundary between audio production and video
production. The system becomes easier to observe and less fragile than a single
monolithic request path.

### 2. Chatterbox should still return useful metadata

`audio_duration_ms` is still worth storing in the job record and returning to
the backend, but MuseTalk should scale from MuseTalk-side pressure and
MuseTalk-side capacity.

## Recommended System Responsibilities

### React Native app

The app should only:

- upload media with presigned S3 URLs
- call the backend to create jobs
- poll or subscribe to job status
- play returned audio or video URLs

The app should not:

- know Vast instance IPs
- know mapped Vast ports
- know worker IDs
- choose Chatterbox or MuseTalk workers
- retry directly against workers

### EC2 backend

The backend should own:

- job creation
- queueing
- worker registry
- worker selection
- retries
- autoscaling
- drain coordination
- S3 handoff
- job state persistence

### Chatterbox-fastest workers

Each Chatterbox worker should:

- boot on a fixed internal app port
- discover its external mapped Vast endpoint
- register itself with the EC2 backend
- expose `/health` and `/ready`
- expose a single synthesis endpoint such as `/tts`
- emit heartbeat data every 15 to 30 seconds
- support drain mode

### MuseTalk workers

Each MuseTalk worker should follow the same worker contract where possible.

This repo already has a boot-oriented Vast path in
`docs/vast_ai_boot.md`, and the current MuseTalk server already exposes useful
telemetry endpoints like `/health`, `/stats`, `/sessions/live`, and
`/hls/sessions/stats`. Those are good building blocks for the future registry
and autoscaler.

## Recommended Worker Contract

Keep the worker contract the same for both pools so the EC2 control plane can
treat them uniformly.

### Required registration payload

```json
{
  "worker_id": "chatterbox-33678878",
  "worker_type": "chatterbox",
  "vast_instance_id": "33678878",
  "endpoint_url": "http://154.61.62.158:50401",
  "public_ip": "154.61.62.158",
  "public_port": 50401,
  "internal_port": 8000,
  "gpu_type": "RTX 3090",
  "status": "ready",
  "max_concurrency": 2,
  "active_jobs": 0,
  "version": "2026-03-29.1"
}
```

### Required heartbeat payload

```json
{
  "worker_id": "chatterbox-33678878",
  "status": "ready",
  "active_jobs": 0,
  "draining": false,
  "queue_wait_ms_p95": 0,
  "avg_job_ms": 2100
}
```

### Recommended Chatterbox response payload

```json
{
  "audio_url": "s3://bucket/jobs/job-123/tts.wav",
  "audio_duration_ms": 4200,
  "voice_id": "female_en_01",
  "sample_rate": 24000
}
```

The `audio_duration_ms` field is useful for job records and downstream
processing, but it should not replace MuseTalk's own capacity signals.

## Queue Design

Separate the work into at least two queues:

- `tts_jobs`
- `avatar_render_jobs`

Optional:

- `preview_tts_jobs`
- `live_avatar_jobs`

Recommended pipeline:

1. Backend receives avatar request.
2. Backend creates a parent job row.
3. Backend enqueues a TTS task to `tts_jobs`.
4. Chatterbox finishes and stores audio in S3.
5. Backend records `audio_duration_ms`.
6. Backend enqueues a MuseTalk render task to `avatar_render_jobs`.
7. MuseTalk downloads from S3 and renders.
8. MuseTalk uploads final result to S3.
9. Backend marks the job complete.

This makes retries and observability much simpler than passing large media
between workers directly.

## Autoscaling Strategy

## Chatterbox pool

Chatterbox should be the first autoscaled pool because it is the easiest to run
as a stateless service.

Suggested starting settings:

- minimum workers: `1`
- warm idle target: `1`
- maximum workers: `3`
- concurrency per worker: start with `2`
- heartbeat interval: `15s`
- scale-up cooldown: `60s`
- scale-down idle timeout: `10m`

Suggested scale-up signals:

- queued TTS jobs >= `2` for `30s`
- no ready Chatterbox worker
- p95 TTS queue wait above target

Suggested scale-down rule:

- worker has `active_jobs = 0`
- worker has been idle for at least `10m`
- healthy worker count is above minimum

## Suggested Desired Worker Formulas

### Chatterbox

```text
desired_chatterbox_workers =
  clamp(
    min_tts_workers,
    max_tts_workers,
    ceil(queued_tts_jobs / tts_jobs_per_worker_target) + tts_warm_idle_target
  )
```

### MuseTalk

MuseTalk should use its own autoscaling document and its own capacity signals.
See `docs/musetalk_autoscaling_plan.md`.

## Routing Rules

### Chatterbox

Worker routing can be simple:

- healthy only
- not draining
- `active_jobs < max_concurrency`
- least in-flight wins

### MuseTalk

For non-live render jobs:

- healthy only
- not draining
- least loaded wins
- optionally prefer a worker with matching cached avatar if you add that signal

For live sessions:

- create session through the backend
- pin session to one MuseTalk node
- store `session_id -> worker_id` in Redis or Postgres
- route all later session calls to the same worker

This pinned-session rule matters much more for MuseTalk than for Chatterbox.

## Why Chatterbox Is A Good First Autoscaling Milestone

Implementing Chatterbox autoscaling first gives you:

- a working worker registry
- a working heartbeat path
- a working drain lifecycle
- a real autoscaler loop
- a real queue
- real per-worker metrics
- less risk than starting with live MuseTalk sessions

After that, MuseTalk scaling becomes a control-plane extension instead of a
greenfield project.

## React Native Contract

The React Native app should continue to talk to one backend contract.

Recommended app-facing endpoints:

- `POST /avatar/jobs`
- `GET /avatar/jobs/:jobId`
- `GET /avatar/jobs/:jobId/result`
- `POST /tts/preview`
- `GET /voices`
- `POST /uploads/presign`

The app should never need to know:

- whether Chatterbox was scaled from 1 worker to 3 workers
- whether MuseTalk was scaled from 1 worker to 5 workers
- which Vast port was assigned externally

That abstraction boundary is the whole point of the EC2 control plane.

## Recommended Rollout Order

### Phase 1

- add Chatterbox worker registration
- add Chatterbox heartbeat
- add Chatterbox drain mode
- add `tts_jobs` queue
- add simple Chatterbox autoscaler

### Phase 2

- store `audio_duration_ms`
- add dashboards and alerts
- keep Chatterbox scaling independent

## Final Recommendation

Yes, Chatterbox should be part of the overall worker-platform implementation,
but it should scale independently from MuseTalk.

The best way to use it is:

- build the worker-control-plane around Chatterbox first
- make Chatterbox emit useful job metadata
- keep MuseTalk autoscaling based on MuseTalk-side pressure and capacity

That gives your React Native app a stable API while letting the backend scale
both pools independently.
