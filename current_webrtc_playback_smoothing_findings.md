# WebRTC Playback Smoothing Findings

Date: 2026-05-18

This note captures the current analysis for reducing WebRTC jitter, skipped
frames, low apparent framerate, and audio speed changes. It is meant to sit
next to `current_webrtc_implementation_plan.md` and update the WebRTC retest
plan with the latest media-path findings.

## User-Visible Symptoms

- The player can report a lower decoded/rendered FPS than the requested target.
  One observed overlay showed `fps target: src=20 out=20` but `video fps: 16.0`.
- Playback looks jittery, with uneven frame cadence and occasional visual
  skipping.
- Audio sometimes sounds like it speeds up or slows down depending on stream
  behavior.
- The desired behavior is closer to HLS: generated frames should be queued and
  played out smoothly, while audio remains at normal speed.

## Active WebRTC Path

The active WebRTC implementation is still independent from the HLS scheduler:

- `api_server.py`
  - `POST /webrtc/sessions/create`
  - `POST /webrtc/sessions/{session_id}/offer`
  - `POST /webrtc/sessions/{session_id}/ice`
  - `POST /webrtc/sessions/{session_id}/stream`
  - `GET /webrtc/player/{session_id}`
- `scripts/webrtc_manager.py`
  - owns `RTCPeerConnection`, session state, the switchable video track, and
    the silence audio track.
- `scripts/webrtc_tracks.py`
  - owns idle/live video playout, live-frame queueing, adaptive FPS, audio
    playout, and A/V sync correction.
- `templates/webrtc_player.py`
  - owns browser/WebView negotiation, remote track attachment, and debug stats.

The stream endpoint starts a direct generation worker in `api_server.py`:

```text
api_server.py
  webrtc_stream()
    streaming_worker()
      avatar.inference_streaming(..., frame_callback=frame_callback, emit_chunks=False)
```

That means WebRTC does not currently inherit the HLS shared GPU scheduler,
ordered segment queueing, startup chunking, HLS status telemetry, or player
buffering semantics.

## Evidence From Recent Logs

Recent WebRTC logs show the playback track deliberately changing effective FPS:

```text
Queue: 18/100 (18%), slowdown: 1.20x, effective_fps: 16.6, played: 30, gen_complete: False
```

This lines up with the screenshot where the browser reports roughly `16.0` FPS
while the target is `20`.

For short clips, generation itself was fast enough:

```text
Total frames generated: 84
Total generation time: 3.67s
Average FPS: 22.88
```

So the 16 FPS symptom is not only model throughput. The WebRTC playout layer is
actively reducing playout FPS when the queue is below its target fill.

For longer clips, the queue can fill completely and the adaptive code speeds
video slightly:

```text
Queue: 100/100 (100%), slowdown: 0.91x, effective_fps: 22.0, played: 900, gen_complete: False
```

That confirms the current video path is not a fixed-rate playout clock. It can
slow down and speed up around the target FPS.

## Root Causes

### 1. Video has a queue, but the queue drives playback speed

`SwitchableVideoStreamTrack` in `scripts/webrtc_tracks.py` already has an
`asyncio.Queue`, prebuffer state, and a `max_queue` of `100`. That part is good.

The problem is `_calculate_adaptive_slowdown()` changes frame time based on queue
fill. At low queue fill, it slows down. At high queue fill, it can speed up.

Relevant path:

```text
scripts/webrtc_tracks.py
  SwitchableVideoStreamTrack._calculate_adaptive_slowdown()
  SwitchableVideoStreamTrack._get_adaptive_frame_time()
  SwitchableVideoStreamTrack.recv()
```

This makes WebRTC unlike HLS. HLS buffers media and plays it at normal media
time. The current WebRTC code tries to protect the queue by retiming playback.
That creates perceptible cadence changes.

### 2. Audio is forced to follow the variable video clock

`SyncedAudioStreamTrack.recv()` computes audio/video drift against
`VideoSyncClock`.

When audio is ahead, it sleeps:

```text
if drift > max_audio_lead:
    sleep(drift - max_audio_lead)
```

When audio is behind, it skips PCM audio frames:

```text
elif drift < -max_audio_lag:
    skip = int((-drift - max_audio_lag) / frame_duration)
    _skip_audio_frames(skip)
```

This directly explains the user-visible audio speed changes. Audio is not being
treated as a stable clock. It is being corrected around a video clock whose rate
can change.

Desired behavior: audio should be packetized at a fixed sample rate and normal
speed. If video falls behind, video should duplicate, hold, or drop frames
according to policy. Audio should not be time-stretched or skipped to follow
video queue conditions.

### 3. aiortc video timestamps are probably wrong for non-30 FPS output

`SwitchableVideoStreamTrack.recv()` calls `await self.next_timestamp()`. aiortc's
base `VideoStreamTrack.next_timestamp()` increments timestamps using
`VIDEO_PTIME = 1 / 30`, regardless of this app's configured `playback_fps`.

Local inspection of the installed aiortc package showed:

```text
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 30
VIDEO_TIME_BASE = 1 / 90000
```

So a `20fps` WebRTC session can be paced by local sleeps at 20 FPS while its RTP
timestamps still advance as if it were 30 FPS. Browsers use RTP timestamps and
jitter buffers for playout decisions, so this mismatch can produce uneven frame
rendering even when frames arrive in order.

This should be fixed by replacing `next_timestamp()` usage in the custom tracks
with app-controlled video PTS:

```text
pts_step = round(90000 / playback_fps)
frame.pts = frame_index * pts_step
frame.time_base = Fraction(1, 90000)
```

The timestamp schedule should match the playout schedule.

### 4. `playback_fps > source_fps` can consume source frames too fast

`SwitchableVideoStreamTrack` computes:

```text
source_step = source_fps / output_fps
advance_frames = int(source_accum)
```

That is meant to duplicate frames when output FPS is higher than source FPS.
However `_pop_live_frames()` loops over `max(steps, 1)`, so it always pops at
least one live source frame even when `advance_frames == 0`.

Result: if `source_fps=10` and `playback_fps=15`, the track may consume source
frames at 15 FPS instead of duplicating some frames. That compresses the live
video duration and then audio correction tries to compensate.

This is especially important because older WebRTC docs recommend `fps=10` and
`playback_fps=15`.

### 5. WebRTC startup prebuffer is currently configured too low for smoothness

The current retest path set:

```text
WEBRTC_VIDEO_PREBUFFER_SECONDS=0
WEBRTC_AUDIO_PREBUFFER_SECONDS=0.15
```

That was useful to prove live video appears, but it is not the smoothest playout
profile. Recent logs show live starts after the first frame:

```text
Prebuffer ready: 1 frames buffered, queue: 1/100
Slowing playback: 1.32x
```

Starting live at one queued frame makes the queue sensitive to burstiness in
GPU, compose, event loop, encoder, and network timing. HLS avoids this by
revealing live only after media is buffered enough to survive normal production
jitter.

### 6. WebRTC lacks HLS-style scheduler fairness and instrumentation

HLS uses `scripts/hls_gpu_scheduler.py` with:

- prep overlap for avatar load and audio feature extraction
- startup-first fairness
- ordered chunk/frame buffers
- compose/encode worker pools
- status telemetry for queue and per-stage timings

WebRTC uses one direct `manager.executor` job per stream plus
`manager.gpu_memory.allocate(session.batch_size)`. There is no shared WebRTC
GPU scheduler, no per-session WebRTC status endpoint, and no exposed track queue
stats outside logs.

This makes WebRTC more vulnerable to cadence jitter under load and harder to
debug than HLS.

### 7. Browser-side audio route may add avoidable sync risk

`templates/webrtc_player.py` attaches video to `remoteVideo` but attaches audio
to `remoteAudio` and also routes it through WebAudio:

```text
remoteVideo.srcObject = remoteStream
remoteAudio.srcObject = audioStream
audioSourceNode = audioContext.createMediaStreamSource(stream)
audioSourceNode.connect(audioContext.destination)
```

Using WebAudio can be useful for autoplay unlocks, but it can also introduce a
separate audio path from the video element. For simplest A/V sync, prefer one
remote `MediaStream` attached to one media element when possible, or keep the
separate audio route only as an explicit fallback.

## Desired WebRTC Behavior

The smoother design should be:

- audio runs at normal speed, fixed `48000 Hz`, 20 ms Opus frames
- video is played at a fixed configured playout FPS
- video timestamps match configured playout FPS
- generated frames are queued before live reveal
- when source FPS is lower than playout FPS, repeat/hold frames
- when queue is temporarily low, hold the last video frame or show idle fallback
- when queue is too high, optionally drop old video frames only if low latency is
  more important than complete playback
- never skip audio to catch up to video
- never speed up or slow down audio because the video queue changed

For this use case, audio should be the master clock. Video should adapt visually
around it.

## Recommended Implementation Plan

### Phase 1: Stabilize existing WebRTC playout

This is the smallest change set likely to fix the obvious symptoms.

1. Disable adaptive FPS by default.

   Change default `WEBRTC_ADAPTIVE_FPS` to false, or start the server with:

   ```bash
   WEBRTC_ADAPTIVE_FPS=0
   ```

   The track should play at a fixed `playback_fps`.

2. Add app-controlled RTP video timestamps.

   Stop using aiortc's hardcoded 30 FPS `next_timestamp()` for custom tracks.
   Use the configured output FPS to set PTS and `time_base`.

3. Fix live-frame consumption when `playback_fps > source_fps`.

   `_pop_live_frames(0)` must not pop a new source frame. It should return the
   previous live frame so output frames can duplicate/hold correctly.

4. Remove audio skip-based drift correction.

   In `SyncedAudioStreamTrack`, do not call `_skip_audio_frames()` for normal
   drift. Audio should emit fixed-size frames at normal sample time. If drift
   telemetry is kept, it should log only at first.

5. Increase initial video prebuffer.

   Recommended first test:

   ```bash
   WEBRTC_VIDEO_PREBUFFER_SECONDS=1.0
   WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0
   WEBRTC_ADAPTIVE_FPS=0
   ```

   At `20fps`, this means roughly 20 frames before live reveal.

6. Keep audio start tied to live reveal, not to adaptive video progress.

   Start audio once the video queue crosses the prebuffer threshold and the
   track is actually switching to live. After start, audio should run normally.

### Phase 2: Make WebRTC more HLS-like

Build an in-memory WebRTC playout buffer that behaves like HLS segments without
writing media segments:

- frame records should carry `source_index`, `playout_index`, `pts`, and
  optional generation timing
- a live stream should move through states: `preparing`, `prebuffering`,
  `playing`, `draining`, `idle`
- the player should not reveal live until the server-side queue reaches a
  minimum playout buffer
- queue low-water and high-water behavior should be explicit:
  - low-water: hold last frame / pause video progression while audio continues
  - high-water: keep all frames for buffered playback, or drop old video frames
    only in a separate low-latency mode

This can be implemented first in `scripts/webrtc_tracks.py` without changing the
GPU scheduler.

### Phase 3: Share HLS scheduler logic with WebRTC

The durable architecture is a shared frame scheduler:

- reuse HLS prep logic for avatar load, audio feature extraction, conditioning,
  and startup fairness
- for HLS, the sink encodes chunks and appends manifest entries
- for WebRTC, the sink pushes composed frames into a timestamped playout queue
- expose identical scheduler metrics for both

This avoids two separate inference pipelines with different fairness and
buffering behavior.

## Suggested Code-Level Changes

### `scripts/webrtc_tracks.py`

- Add a custom video timestamp method using `self._output_fps`.
- Replace `time.time()` with `time.monotonic()` for playout pacing.
- Disable adaptive FPS by default.
- Change `_pop_live_frames(steps)` so `steps <= 0` does not consume a new frame.
- Add explicit duplicate-frame counting to stats.
- Add `queue_low_water_frames`, `queue_high_water_frames`, and
  `prebuffer_frames` stats.
- Change audio drift correction from "sleep/skip" to "measure/log".
- Make audio playout fixed at `samples_per_frame / sample_rate`.

### `api_server.py`

- Make WebRTC defaults favor stable playback:

  ```text
  WEBRTC_VIDEO_PREBUFFER_SECONDS=1.0
  WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0
  WEBRTC_ADAPTIVE_FPS=0
  ```

- Add `GET /webrtc/sessions/{session_id}/status`.
- Include track stats:
  - queue size/max/fill
  - prebuffer readiness
  - frames received
  - frames played
  - frames duplicated
  - frames dropped
  - generation complete
  - configured FPS
  - effective playout FPS
  - audio frames sent
  - audio duration sent
  - measured A/V drift

### `templates/webrtc_player.py`

- Add more useful `getStats()` overlay fields:
  - `framesReceived`
  - `framesDecoded`
  - `framesDropped`
  - `jitterBufferDelay`
  - `jitterBufferEmittedCount`
  - `freezeCount`
  - `totalFreezesDuration`
  - audio `jitterBufferDelay`
  - audio concealed samples, if available
- Prefer a single remote stream attached to one media element when possible.
- Keep WebAudio route as a fallback for autoplay/audio unlock issues.

### Docs / Run Scripts

Update WebRTC launch defaults in `scripts/run_webrtc_relay_api_server.sh` after
code fixes are in place:

```bash
export WEBRTC_VIDEO_PREBUFFER_SECONDS="${WEBRTC_VIDEO_PREBUFFER_SECONDS:-1.0}"
export WEBRTC_AUDIO_PREBUFFER_SECONDS="${WEBRTC_AUDIO_PREBUFFER_SECONDS:-0.0}"
export WEBRTC_ADAPTIVE_FPS="${WEBRTC_ADAPTIVE_FPS:-0}"
```

## Recommended Test Matrix

Start with one stream and stable local/browser playback:

```bash
POST /webrtc/sessions/create?fps=20&playback_fps=20&batch_size=4&chunk_duration=1
POST /webrtc/sessions/{id}/stream
```

Expected behavior:

- browser FPS should stay close to 20 after prebuffer
- no audio speedup/slowdown
- queue starts around prebuffer target and drains at fixed rate
- no video speedup after generation completes

Then test source/output mismatch:

```bash
POST /webrtc/sessions/create?fps=10&playback_fps=20&batch_size=4&chunk_duration=1
```

Expected behavior:

- video duration should match audio duration
- every source frame should be duplicated roughly once
- audio should not skip

Then test insufficient generation throughput:

```bash
POST /webrtc/sessions/create?fps=24&playback_fps=24&batch_size=8&chunk_duration=1
```

Expected behavior:

- if generation cannot sustain 24 FPS, video should hold or drop according to
  policy
- audio should remain normal speed
- telemetry should make the queue underrun obvious

Finally test public/mobile:

- TURN relay enabled
- `WEBRTC_ICE_TRANSPORT_POLICY=relay`
- HTTPS for real iOS/Android devices
- compare local network vs cellular

## Operational Recommendation For Next Retest

Use a stable profile first, then tune for latency:

```bash
WEBRTC_ADAPTIVE_FPS=0
WEBRTC_VIDEO_PREBUFFER_SECONDS=1.0
WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0
WEBRTC_AUDIO_MAX_LEAD_SECONDS=0.0
WEBRTC_AUDIO_MAX_LAG_SECONDS=0.0
```

The lead/lag values should become irrelevant once skip-based correction is
removed. Until then, setting them to zero is not enough because the current code
will still react to drift; the code path itself should be changed.

## Bottom Line

The current WebRTC jitter is not just network jitter. The server intentionally
changes video playout speed based on queue depth, then the audio track corrects
itself against that variable video clock by sleeping or skipping samples. On top
of that, custom video tracks appear to inherit aiortc's default 30 FPS timestamp
step even when the session is configured for 20 FPS.

To make WebRTC feel like HLS, keep a real queue but remove variable-rate media
playout. Use audio as the stable clock, timestamp video at the configured
playout FPS, prebuffer before reveal, and adapt video with duplication/holding
instead of changing audio speed.
