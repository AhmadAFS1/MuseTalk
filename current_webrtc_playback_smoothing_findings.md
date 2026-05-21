# WebRTC Playback Smoothing Findings

Date: 2026-05-18

This note captures the current analysis for reducing WebRTC jitter, skipped
frames, low apparent framerate, and audio speed changes. It is meant to sit
next to `current_webrtc_implementation_plan.md` and update the WebRTC retest
plan with the latest media-path findings.

## Implementation Update: 2026-05-21 Shared WebRTC Generation

The latest wall investigation found that the browser was not the reason four
WebRTC streams appeared to take turns. One browser tab can hold multiple WebRTC
peer connections, and the player now attaches the remote audio and video tracks
to one combined `MediaStream` on a single `<video>` element. The remaining
turn-taking came from the backend: each WebRTC stream used its own generation
worker and leased `session.batch_size` GPU capacity independently. With the
current `batch_size=8` throughput profile, those workers effectively serialized
behind the GPU lease/memory budget, so the playout gates were released roughly
one clip apart instead of together.

The HLS wall does not have that problem because all active HLS jobs enter
`HLSGPUStreamScheduler`, which builds one shared GPU batch across sessions and
then fans the composed results back out to each stream. WebRTC now uses that same
shared scheduler by default:

- `scripts/hls_gpu_scheduler.py`
  - added `output_mode="webrtc"` jobs
  - added `submit_webrtc_stream(...)`
  - keeps avatar load, audio feature extraction, Whisper conditioning,
    scheduler startup fairness, GPU batching, and compose workers shared with
    HLS
  - sends composed frames directly to the WebRTC frame callback instead of
    encoding HLS segments
  - keeps WebRTC jobs in startup priority until the initial startup frame block
    is produced, instead of marking them warmed after one frame
- `api_server.py`
  - `POST /webrtc/sessions/{session_id}/stream` now defaults to
    `WEBRTC_SHARED_GPU_SCHEDULER=1`
  - group `Start All` inherits this because it calls the same per-session stream
    route for every session
  - the old independent WebRTC generation worker remains available with
    `WEBRTC_SHARED_GPU_SCHEDULER=0`
  - scheduler completion signals `SwitchableVideoStreamTrack` generation
    completion, releases short clips if needed, and lets playback drain before
    returning the session to idle

This does not make WebRTC a literal single audio+video RTP track. WebRTC still
uses separate RTP audio and video tracks, because that is the normal browser
media model. The HLS similarity is at two important layers:

- browser playback: one `<video>` element receives a combined remote
  `MediaStream`, so native media-element A/V sync can apply
- backend production: all active streams share one scheduler and one GPU batch
  loop, so concurrent wall streams should generate together instead of being
  serialized by per-session GPU leases

Audio source clarification: the uploaded audio is the authoritative audio track.
MuseTalk generates video frames from that audio; the generated frame/chunk output
is not an MP4 that already contains final audio. HLS later muxes generated video
frames with the uploaded audio into HLS segments. WebRTC sends the uploaded audio
through `SyncedAudioStreamTrack` and sends generated video frames through
`SwitchableVideoStreamTrack`, then the browser receives both tracks in one media
element.

There is no intentional static A/V offset in this path. The only configured
startup delay is `WEBRTC_AV_START_DELAY_SECONDS` (default `0.05`), which is a
shared release delay applied to both tracks after audio preparation and video
readiness. It is not an audio-only or video-only offset.

Wall/player updates from this pass:

- `templates/webrtc_player.py` now supports debug display modes:
  `debug=docked`, `debug=overlay`, and `debug=off`.
- The wall exposes a stats/debug toggle so the ICE/FPS/audio counters can be
  viewed without covering the video.
- The wall has `Connect All` for attaching all iframes/peer connections before
  `Start All` uploads audio. `Start All` starts generation; `Connect All` only
  establishes/warms the browser WebRTC connections.

HLS wall reference data for matching WebRTC tests is in
`current_cross_server_throughput_findings.md` under "Full-rate `20/20` FPS
validation on May 10, 2026":

- `20/20`, request `batch_size=8`, current `8,16` throughput profile,
  `concurrency=4`: completed `4/4`, `avg_time_to_live_ready_s=1.889`,
  `avg_segment_interval_s=1.188`, `max_segment_interval_s=2.041`,
  `wall_time_s=22.4`.
- `20/20`, request `batch_size=8`, current `8,16` throughput profile,
  `concurrency=5`: completed `5/5`, `avg_time_to_live_ready_s=2.014`,
  `avg_segment_interval_s=1.477`, `max_segment_interval_s=2.550`,
  `wall_time_s=27.5`.

Recommended WebRTC wall retest after restarting:

```bash
WEBRTC_SHARED_GPU_SCHEDULER=1
WEBRTC_SYNC_MODE=strict_fifo
WEBRTC_ADAPTIVE_FPS=0
WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0
WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0
```

Use `count=4`, `fps=20`, `playback_fps=20`, `batch_size=8`,
`chunk_duration=1`, click `Connect All`, then `Start All`. Expected logs should
show every stream queued for the shared `WEBRTC` GPU scheduler and the first
startup blocks/release gates occurring close together, not one full clip apart.

## Implementation Update: 2026-05-18

Phase 1 items 1-6 have now been implemented.

### A/V Start Barrier Update: 2026-05-18

The follow-up sync fix is now implemented. The main issue found in the logs was
not dropped video frames or WebRTC network jitter; it was a startup race where
live video could begin consuming the FIFO as soon as video prebuffer was ready,
while the audio track was still converting/loading through FFmpeg or PyAV.

New behavior:

- `SyncedAudioStreamTrack.prepare()` converts/decodes audio before live playout
  is released.
- `api_server.py` starts audio preparation immediately after replacing the audio
  track, in parallel with video generation.
- `VideoSyncClock` now owns a shared A/V playout gate:
  - video prebuffer marks `video_ready`
  - audio preparation marks `audio_ready`
  - the server releases `playout_released` only after both are ready
  - a short configurable `WEBRTC_AV_START_DELAY_SECONDS` delay, default `0.05`,
    gives both tracks a common release point without changing audio speed
- `SwitchableVideoStreamTrack.recv()` keeps showing idle frames after prebuffer
  readiness until the shared playout gate is due. It does not consume the first
  live FIFO frame early.
- `SyncedAudioStreamTrack.recv()` waits for the same playout gate and then waits
  for video coverage as before. Audio samples are still emitted at fixed
  48 kHz / 20 ms cadence; they are not resampled, sped up, slowed down, or
  skipped.
- New telemetry is exposed in `/webrtc/sessions/{session_id}/status` and
  `/webrtc/sessions/stats`, including:
  - `audio.prepare_seconds`
  - `audio.first_packet_after_signal_seconds`
  - `sync_clock.audio_ready`
  - `sync_clock.video_ready`
  - `sync_clock.playout_released`
  - `sync_clock.first_video_frame_after_release_seconds`
  - `sync_clock.first_audio_packet_after_release_seconds`
  - `sync_clock.initial_av_start_delta_seconds`

Expected result: differences between audio files should no longer create a
variable 90-200 ms initial lip-sync offset, because audio conversion/loading is
absorbed before the live FIFO is released.

Changed files:

- `scripts/webrtc_tracks.py`
  - added the shared `playout_released` start barrier to `VideoSyncClock`
  - added audio preparation telemetry and first packet/frame release telemetry
  - holds live video behind the A/V gate after prebuffer readiness so the first
    live FIFO frame cannot outrun audio setup
  - changed default `WEBRTC_ADAPTIVE_FPS` to off
  - changed default `WEBRTC_SYNC_MODE` to `strict_fifo`
  - changed default `WEBRTC_VIDEO_PREBUFFER_SECONDS` to `2.0`
  - changed default live queue depth to `400` frames in strict FIFO mode
  - replaced aiortc's default 30 FPS video timestamp helper with fixed RTP
    timestamps based on the configured output FPS
  - fixed live frame consumption so `playback_fps > source_fps` duplicates/holds
    frames instead of consuming source frames too fast
  - added strict FIFO A/V gating so audio waits for emitted video timeline
    coverage and video waits for the next FIFO frame instead of skipping/holding
    during underrun
  - removed skip/sleep audio drift correction; audio now emits fixed 48 kHz
    20 ms frames after start
  - starts the audio playout clock only after audio is loaded and ready to emit,
    and re-anchors it after strict FIFO stalls to avoid catch-up bursts
  - added stats for duplicated frames, output frames, underruns, stalls,
    source/output FPS, sync mode, and audio playout state
- `api_server.py`
  - prepares audio immediately after replacing the WebRTC audio track
  - releases A/V playout only after audio preparation and video prebuffer are
    both ready
  - adds configurable `WEBRTC_AV_START_DELAY_SECONDS`, default `0.05`, for a
    shared release point
  - starts live video before queueing the first generated frame
  - signals audio only after the video track reports prebuffer readiness
  - releases queued frames and starts audio for short clips that finish before
    the configured prebuffer fills
  - adds `GET /webrtc/sessions/{session_id}/status` with video/audio/sync stats
  - increases default WebRTC frame handoff timeout in strict FIFO mode so queue
    backpressure preserves frame order instead of dropping frames
- `templates/webrtc_player.py`
  - attaches remote audio and video tracks to the same `MediaStream` on the
    video element so the browser can apply native A/V sync
- `scripts/run_webrtc_relay_api_server.sh`
  - defaults WebRTC relay launches to `WEBRTC_SYNC_MODE=strict_fifo`,
    `WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0`, `WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0`,
    and `WEBRTC_ADAPTIVE_FPS=0`

## A/V Sync Follow-Up: Strict FIFO Mode

After restarting with the Phase 1 changes, video playback could look much
smoother while A/V sync still felt fragile. The strict FIFO pass preserves the
user's preferred behavior: WebRTC remains FIFO-like and does not skip generated
video frames, while audio does not speed up or slow down.

That combination is possible only if WebRTC is allowed to behave like HLS when
production falls behind: buffer or stall both media tracks together. If we require
all generated video frames to be shown and audio to remain normal speed, then the
system cannot also promise continuous low-latency playout during a video
underrun. In that case, the correct sync-preserving behavior is:

- keep video frames in FIFO order
- keep audio samples in FIFO order
- start both only after enough shared A/V buffer exists
- if the next video frame is not ready, stop advancing audio too
- resume both from the same media timestamp once video catches up
- never solve drift by speeding audio, slowing audio, skipping audio, or skipping
  generated video frames

This is closer to HLS buffering than normal ultra-low-latency WebRTC. The user
experience should be a short buffering/stall event rather than lip sync drift.

The implementation now uses a shared A/V playout gate instead of letting audio
and video tracks free-run:

```text
Generated video FIFO:
  frame_index -> media_time = frame_index / source_fps

Decoded audio FIFO:
  packet_index -> media_time = samples_sent / sample_rate

Shared playout state:
  next_audio_packet
  next_video_frame
  buffered_video_until
  buffered_audio_until
  playout_started
```

In strict mode, `SyncedAudioStreamTrack.recv()` and
`SwitchableVideoStreamTrack.recv()` both consult the same playout state. Audio
emits packet `N` only when the corresponding video timeline coverage exists.
Video emits frame `N` in FIFO order at the configured output cadence. If either
side is missing, both tracks wait. This makes sync forced by the server's media
queue, not inferred after the fact from drift correction.

This is now the default mode and can be set explicitly:

```bash
WEBRTC_SYNC_MODE=strict_fifo
WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0
WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0
WEBRTC_ADAPTIVE_FPS=0
```

The tradeoff is latency and occasional buffering. That is acceptable for this
mode because the priority is complete FIFO playback and stable lip sync.

## A/V Sync Retest: 2026-05-20

After fixing the WebRTC batch-size mismatch so the `throughput_record` profile
uses warmed TRT batches (`8,16`), WebRTC video playback became smooth, but A/V
sync still failed in the expected direction: audio finished before video.

Observed run:

```text
Source audio duration: 15.170958s
Total frames generated: 303
Generation FPS setting: 20
Video media duration: 303 / 20 = 15.15s
Generation complete signaled. Queue: 100, Played: 203
Audio EOF after 758 frames
Playback complete - returning to idle
Live mode ended. Played: 303, Dropped: 0, Drained: 0
Audio/video drift observed: 0.450s, 0.650s, 0.850s, 1.050s, 1.200s, 1.350s
```

The audio and video media lengths were effectively the same, so this was not an
asset-duration mismatch. The drift came from playout policy:

- audio starts once the video prebuffer threshold is reached
- audio then free-runs at fixed `48000 Hz` / 20 ms packets
- video drains a generated-frame FIFO at `playback_fps`
- when generation outruns playout, video can still have a large queue after
  generation completes
- audio reaches EOF while video is still draining queued frames

In that run, audio reached EOF when video had played about `288/303` frames,
leaving roughly `15` video frames, or `0.75s` at 20 FPS, after the spoken audio.
The drift telemetry reached about `1.35s`.

This confirms that the current `VideoSyncClock` is not a sync mechanism. It
gates audio start and records drift, but it does not stop audio from advancing
past missing or delayed video timeline coverage:

```text
scripts/webrtc_tracks.py
  SyncedAudioStreamTrack.recv()
    waits for _started
    waits for sync_clock.started
    emits fixed audio packets from its own monotonic start time
    logs drift only
```

Before the 2026-05-20 combined-stream experiment, the WebRTC browser player
also did not mirror HLS playback semantics:

```text
templates/webrtc_player.py
  remoteVideo.srcObject = remoteStream           # video track
  remoteAudio.srcObject = new MediaStream(...)   # separate audio track
  WebAudio route = createMediaStreamSource(audioStream)
```

That meant the browser was not using one media element as the primary A/V
playout surface. The WebAudio route is useful for autoplay unlock experiments,
but it adds a separate audio path and should not be the default sync path.

Implemented experiment on 2026-05-20:

```text
templates/webrtc_player.py
  <video id="remoteVideo" ...>
  remoteStream = new MediaStream()
  pc.ontrack -> remoteStream.addTrack(event.track)   # audio and video
  remoteVideo.srcObject = remoteStream
```

This makes the WebRTC player use one browser media element for both remote RTP
tracks, which is the closest client-side equivalent to the HLS player. If audio
still finishes before video after this change, the remaining cause is server
playout policy: audio is still emitted from its own `SyncedAudioStreamTrack`
clock while video drains its own queue.

### Can WebRTC Use One Track Like HLS?

Not literally. HLS works by muxing encoded audio and video into the same media
segment. The browser plays that muxed segment with one `<video>` element, so one
media timeline owns both audio and video timestamps.

WebRTC media is different: audio and video are separate RTP media tracks
(`m=` sections) even when they are in the same peer connection. A WebRTC video
track cannot carry audio samples, and an audio track cannot carry video frames.
So "one track containing audio and video" is not a WebRTC-native option.

What is possible, and what we should do, is make WebRTC behave like one media
timeline:

- keep one `RTCPeerConnection`
- put audio and video into the same remote `MediaStream` on the client
- attach that combined stream to one `<video>` element as the default path
- keep any separate `<audio>` / WebAudio route out of the default sync path
- on the server, introduce a shared A/V playout gate so audio and video cannot
  advance independently

The server-side gate is the important part. Browser-side stream attachment can
reduce avoidable sync risk, but it cannot fix audio finishing early if the
server continues to emit audio while video is still queued behind.

The HLS-equivalent WebRTC design is therefore "two RTP tracks, one playout
timeline", not "one RTP track". In strict FIFO mode:

```text
video frame N -> media_time = N / fps
audio packet M -> media_time = samples_sent / sample_rate

Audio packet M may be emitted only if video timeline coverage has reached the
same media time. If video coverage is missing, both audio and video stall. When
video catches up, both resume from the same media timestamp.
```

This intentionally trades latency for lip-sync correctness. It is the correct
mode when the product requirement is "do not skip generated video frames and do
not speed/skip audio".

## Tests Performed: 2026-05-18

Syntax/compile checks:

```bash
python3 -m py_compile scripts/webrtc_tracks.py scripts/webrtc_manager.py api_server.py templates/webrtc_player.py
/workspace/.venvs/musetalk_trt_stagewise/bin/python -m py_compile scripts/webrtc_tracks.py scripts/webrtc_manager.py api_server.py templates/webrtc_player.py
```

Additional checks after the A/V start barrier update:

```bash
python3 -m py_compile scripts/webrtc_tracks.py api_server.py templates/webrtc_player.py scripts/webrtc_manager.py
/workspace/.venvs/musetalk_trt_stagewise/bin/python -m py_compile scripts/webrtc_tracks.py api_server.py templates/webrtc_player.py scripts/webrtc_manager.py
```

In-process strict FIFO barrier smoke test:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python
```

- Created a strict FIFO `VideoSyncClock`.
- Created `SwitchableVideoStreamTrack(source_fps=10, output_fps=10,
  prebuffer_seconds=0.1)`.
- Pushed one generated frame and verified prebuffer became ready.
- Called `recv()` before releasing the shared A/V gate and verified the live
  frame was not consumed (`frames_played == 0`).
- Prepared `SyncedAudioStreamTrack` before release and verified audio prep
  telemetry was populated.
- Released the shared playout gate with a small future `t0`.
- Verified the next video `recv()` consumed exactly the first live FIFO frame.
- Verified the first audio `recv()` emitted fixed PTS `0`.
- Verified sync telemetry was populated:

```text
strict_fifo_barrier_ok
initial_av_start_delta_seconds=0.000332878902554512
audio_prepare_seconds=0.056449003983289
```

This proves the server-side race is closed in-process: video prebuffer readiness
alone is no longer enough to start live video, and the measured first live video
frame to first audio packet delta was about `0.3 ms` in the smoke test.

In-process media track smoke tests used the project venv:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python
```

- Created `SwitchableVideoStreamTrack(source_fps=10, output_fps=20,
  prebuffer_seconds=0, adaptive_fps=False)`.
- Pushed three generated frames, marked generation complete, and called `recv()`
  through playback drain.
- Verified video RTP PTS values were spaced by `4500` ticks, which is the
  correct 90 kHz clock step for 20 FPS.
- Verified only `3` source frames were consumed while `3` output frames were
  duplicated, confirming `playback_fps > source_fps` no longer speeds through
  source frames.
- Created `SyncedAudioStreamTrack(..., use_ffmpeg_convert=False)` and called
  `recv()` four times.
- Verified audio PTS values were `[0, 960, 1920, 2880]`, confirming fixed
  48 kHz / 20 ms audio packet timing.
- Created `SwitchableVideoStreamTrack(source_fps=10, output_fps=10,
  prebuffer_seconds=0.5, adaptive_fps=False)`.
- Verified prebuffer readiness stayed false for the first four frames and became
  true on the fifth frame, matching a 0.5 s prebuffer at 10 FPS.
- Created `VideoSyncClock(strict_fifo=True)` with
  `SwitchableVideoStreamTrack(source_fps=10, output_fps=10)`.
- Called `recv()` before pushing a live frame and verified video waited for the
  missing FIFO frame instead of returning a held/idle frame.
- Created `SyncedAudioStreamTrack` with the same strict FIFO clock.
- Verified audio emitted packets through `0.100s` of video coverage, then waited
  before emitting the next packet until the second video frame advanced coverage
  to `0.200s`.
- Verified strict FIFO 10 FPS -> 20 FPS playback still used PTS spacing of
  `4500` ticks, consumed only `3` source frames, duplicated `3` output frames,
  and dropped `0` frames.
- Restarted the API with `WEBRTC_SYNC_MODE=strict_fifo`,
  `WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0`, `WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0`,
  and `WEBRTC_ADAPTIVE_FPS=0`.
- Created a WebRTC session through the live API and verified
  `GET /webrtc/sessions/{session_id}/status` reported `sync_mode=strict_fifo`,
  `prebuffer_frames=40`, `queue_max=400`, and `adaptive_fps=false` for a
  20 FPS / 20 FPS session.

These are smoke tests of the server-side media tracks. They do not prove browser
smoothness by themselves; the next validation should compare browser `getStats()`
and the server status endpoint during a real WebRTC stream.

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

This has now been fixed by replacing `next_timestamp()` usage in the custom
tracks with app-controlled video PTS:

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
`playback_fps=15`. The Phase 1 implementation now holds/duplicates frames for
this case instead of consuming source frames at the output FPS.

### 5. WebRTC startup prebuffer is currently configured too low for smoothness

The old retest path set:

```text
WEBRTC_VIDEO_PREBUFFER_SECONDS=0
WEBRTC_AUDIO_PREBUFFER_SECONDS=0.15
```

That was useful to prove live video appears, but it is not the smoothest playout
profile. Older logs showed live starting after the first frame:

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

Before the 2026-05-20 combined-stream experiment, `templates/webrtc_player.py`
attached video to `remoteVideo` but attached audio to `remoteAudio` and also
routed it through WebAudio:

```text
remoteVideo.srcObject = remoteStream
remoteAudio.srcObject = audioStream
audioSourceNode = audioContext.createMediaStreamSource(stream)
audioSourceNode.connect(audioContext.destination)
```

Using WebAudio can be useful for autoplay unlocks, but it can also introduce a
separate audio path from the video element. The player now prefers one remote
`MediaStream` attached to one media element. If the new player still drifts, the
server-side audio/video playout gate is the next fix to test.

## Desired WebRTC Behavior

The smoother design should be:

- audio runs at normal speed, fixed `48000 Hz`, 20 ms Opus frames
- video is played at a fixed configured playout FPS
- video timestamps match configured playout FPS
- generated frames are queued before live reveal
- when source FPS is lower than playout FPS, repeat/hold frames
- in strict FIFO mode, when the video queue is temporarily low, pause audio and
  video progression together instead of letting audio run ahead
- in low-latency mode only, hold the last video frame or drop late video frames
  if continuous audio is more important than complete video playback
- never skip audio to catch up to video
- never speed up or slow down audio because the video queue changed
- never skip generated video frames in strict FIFO mode

For this use case, strict FIFO should use a shared A/V playout clock. Audio is
still fixed-rate once emitted, but it must not be allowed to free-run past missing
video coverage.

## Recommended Implementation Plan

### Phase 1: Stabilize existing WebRTC playout

Status: implemented on 2026-05-18.

1. Disable adaptive FPS by default. Done.

   Change default `WEBRTC_ADAPTIVE_FPS` to false, or start the server with:

   ```bash
   WEBRTC_ADAPTIVE_FPS=0
   ```

   The track should play at a fixed `playback_fps`.

2. Add app-controlled RTP video timestamps. Done.

   Stop using aiortc's hardcoded 30 FPS `next_timestamp()` for custom tracks.
   Use the configured output FPS to set PTS and `time_base`.

3. Fix live-frame consumption when `playback_fps > source_fps`. Done.

   `_pop_live_frames(0)` must not pop a new source frame. It should return the
   previous live frame so output frames can duplicate/hold correctly.

4. Remove audio skip-based drift correction. Done.

   In `SyncedAudioStreamTrack`, do not call `_skip_audio_frames()` for normal
   drift. Audio should emit fixed-size frames at normal sample time. If drift
   telemetry is kept, it should log only at first.

5. Increase initial video prebuffer. Done.

   Recommended first test:

	   ```bash
	   WEBRTC_SYNC_MODE=strict_fifo
	   WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0
	   WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0
	   WEBRTC_ADAPTIVE_FPS=0
	   ```

	   At `20fps`, this means roughly 40 frames before live reveal.

6. Keep audio start tied to live reveal/prebuffer readiness, not to adaptive
   video progress. Done.

   Start audio once the video queue crosses the prebuffer threshold and the
   track is actually switching to live. After start, audio should run normally.

### Phase 2: Make WebRTC more HLS-like

Build an in-memory WebRTC playout buffer that behaves like HLS segments without
writing media segments. The preferred mode is strict FIFO A/V sync:

- frame records should carry `source_index`, `playout_index`, `pts`, and
  optional generation timing
- audio packets should carry `sample_index`, `packet_index`, and media time
- both audio and video should be buffered against the same media timeline
- a live stream should move through states: `preparing`, `prebuffering`,
  `playing`, `draining`, `idle`
- the player should not reveal live until the server-side queue reaches a
  minimum shared A/V playout buffer
- queue low-water and high-water behavior should be explicit:
  - strict FIFO low-water: pause both audio and video progression until the next
    required video frame is available
  - strict FIFO high-water: keep all frames for buffered playback, accepting
    higher latency
  - low-latency mode only: hold/drop video when continuous audio is more
    important than showing every generated frame

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
- Add a strict FIFO sync mode with a shared A/V playout gate.
- Give generated video frames explicit source media time and queue them in FIFO
  order.
- Give audio packets explicit sample/media time and only release packet `N` when
  the corresponding video timeline coverage is ready.
- In strict FIFO mode, make low-water behavior stall both tracks rather than
  letting audio run ahead or skipping video frames.
- Track stall count/duration and current shared buffer depth in stats.

### `api_server.py`

- Make WebRTC defaults favor stable playback:

	  ```text
	  WEBRTC_SYNC_MODE=strict_fifo
	  WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0
	  WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0
	  WEBRTC_ADAPTIVE_FPS=0
	  ```

- Add `GET /webrtc/sessions/{session_id}/status`.
- Include track stats:
  - queue size/max/fill
  - shared A/V buffer depth
  - prebuffer readiness
  - frames received
  - frames played
  - frames duplicated
  - frames dropped
  - A/V stalls and stall duration
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
POST /webrtc/sessions/create?fps=20&playback_fps=20&batch_size=8&chunk_duration=1
POST /webrtc/sessions/{id}/stream
```

Expected behavior:

- browser FPS should stay close to 20 after prebuffer
- no audio speedup/slowdown
- queue starts around prebuffer target and drains at fixed rate
- no video speedup after generation completes
- in strict FIFO mode, audio must not finish before queued video drains

Then test source/output mismatch:

```bash
POST /webrtc/sessions/create?fps=10&playback_fps=20&batch_size=8&chunk_duration=1
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
WEBRTC_SYNC_MODE=strict_fifo
WEBRTC_ADAPTIVE_FPS=0
WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0
WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0
```

`WEBRTC_AUDIO_MAX_LEAD_SECONDS` and `WEBRTC_AUDIO_MAX_LAG_SECONDS` are now only
drift log thresholds. They no longer change audio pacing, so leave them at the
defaults unless the drift logs are too noisy or too quiet.

## Bottom Line

The observed WebRTC jitter was not just network jitter. The server was
intentionally changing video playout speed based on queue depth, then the audio
track corrected itself against that variable video clock by sleeping or skipping
samples. On top of that, custom video tracks inherited aiortc's default 30 FPS
timestamp step even when the session was configured for 20 FPS.

To make WebRTC feel like HLS, keep a real queue but remove variable-rate media
playout. Use audio as the stable clock, timestamp video at the configured
playout FPS, prebuffer before reveal, and adapt video with duplication/holding
instead of changing audio speed.

The 2026-05-20 retest sharpens this conclusion: removing variable video speed
and audio skip/sleep correction is not sufficient. Audio and video still need a
shared playout gate. Without it, audio can free-run to EOF while video is still
draining a FIFO queue, even when both media durations match.
