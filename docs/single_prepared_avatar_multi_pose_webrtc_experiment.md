# Single-Prepared-Avatar Multi-Pose WebRTC Experiment

## Goal

Test whether several highly similar Kling 2.6 motion videos can share one
MuseTalk prepared-avatar bundle while still supporting convincing live lip sync.

The experiment deliberately avoids preparing every pose. Only
`neutral_resting` is prepared. The other pose videos are decoded during the
WebRTC session and used as alternate composition backgrounds for MuseTalk's
generated face output.

## Hypothesis

The three poses are similar enough in framing, face scale, head angle, and body
position that MuseTalk inference based on the prepared `neutral_resting` cycle
can be composited onto `speaking_direct` and `light_smile` frames without major
drift.

If the hypothesis holds:

- lip-sync inference performance should remain close to the existing
  single-avatar path
- GPU memory should remain close to one prepared avatar
- extra memory should be limited to video decoder state and a small number of
  active frames
- no per-pose latent, mask, coordinate, or full-frame prepared bundle should be
  required

## Source Motion References

Use these existing reference clips:

| Runtime role | Reference file |
| --- | --- |
| Canonical idle and only prepared source | `assets/processed_videos_short/neutral_resting.mp4` |
| Periodic expression during idle and speech | `assets/processed_videos_short/light_smile.mp4` |
| Primary background while speaking | `assets/processed_videos_short/speaking_direct.mp4` |

All three current reference clips are approximately 4.9 seconds, 720x1280,
30 FPS, and H.264.

## Asset Generation

Generate three Kling 2.6 Motion Control outputs from the same avatar image:

1. `neutral_resting`
2. `light_smile`
3. `speaking_direct`

Use the corresponding processed video as each Kling motion reference. Keep the
avatar image, output dimensions, camera framing, character orientation, and
generation settings identical across all three requests.

The Segmind API credential must be supplied through an environment variable for
the generation process. Do not write it to this document, source code, fixtures,
shell scripts, metadata, logs, or committed environment files.

## Preparation Rule

Prepare only the generated `neutral_resting` video as the MuseTalk avatar.

Do not create independent prepared-avatar assets for `light_smile` or
`speaking_direct`. In particular, do not create or retain additional per-pose:

- VAE latent cycles
- full frame cycles
- face mask cycles
- face coordinate cycles
- compose-plan cycles

The generated `light_smile` and `speaking_direct` MP4s are lightweight runtime
pose sources attached to the WebRTC session.

## Experimental Live Composition Path

The existing MuseTalk path selects inference latents and composition backgrounds
from the same prepared source cycle. This experiment separates those choices.

During live speech:

1. Continue selecting MuseTalk inference latents from the prepared
   `neutral_resting` cycle.
2. Decode the active pose MP4 incrementally inside the WebRTC session.
3. Use the active pose frame as the composition background instead of the
   prepared neutral full frame.
4. Initially reuse the neutral cycle's face coordinates, mask, and compose
   geometry at the corresponding normalized frame position.
5. Composite the generated face result onto the active pose frame.
6. Keep the existing neutral composition path available as an immediate
   fallback.

Do not preload every pose video into RAM. Keep only decoder state, transition
frames, and a bounded frame buffer.

## Frame Mapping

Map pose frames to the neutral prepared cycle by normalized progress rather than
assuming identical frame counts:

```text
pose_progress = active_pose_frame_index / active_pose_frame_count
neutral_index = round(pose_progress * neutral_cycle_frame_count)
```

Clamp the result to a valid neutral cycle index. Preserve the current ping-pong
cycle behavior where required by MuseTalk.

The first test intentionally reuses neutral geometry. Do not add per-pose
preparation before measuring whether this simple mapping is sufficient.

## WebRTC Behavior

Use one persistent WebRTC video track. Pose changes must not replace the sender
or require SDP renegotiation.

Target sequence:

```text
session starts
neutral_resting idle
light_smile idle flourish
neutral_resting idle
TTS starts
speaking_direct live composition
light_smile live composition
speaking_direct live composition
TTS ends
neutral_resting idle
```

Idle pose changes can continue using the existing idle-video switch path. Live
pose changes require the experimental alternate-background composition path.

Use a short crossfade or transition-frame queue when changing pose sources.
Start with the existing 0.35-second WebRTC idle transition duration as the test
default. Do not delay speech while waiting for a pose transition.

## Audio Fixtures

Use existing repository audio before generating a new TTS fixture:

| Test | Audio | Purpose |
| --- | --- | --- |
| Short smoke test | `data/audio/yongen.wav` | About 8 seconds; exercises speech start, live output, trailing silence, and return to idle |
| Long transition test | `data/audio/eng.wav` | About 60 seconds; allows repeated live pose changes and sustained performance measurement |

The short test should run first. A custom TTS sample can be added later if the
existing speech does not provide enough duration or cadence for judging the
transitions.

## Test Matrix

### Baseline

Run `neutral_resting` as both the prepared source and live composition
background. Record timing, memory, and a screen capture.

### Alternate Speaking Background

Keep neutral inference latents but compose onto `speaking_direct`. Compare mouth
alignment, face identity, head stability, and throughput against baseline.

### Alternate Smile Background

Keep neutral inference latents but compose onto `light_smile`. Check whether the
generated mouth preserves the smile naturally or produces jaw, cheek, or mask
artifacts.

### Full Sequence

Run the complete idle-to-speaking-to-smile-to-speaking-to-idle sequence using
the short audio fixture. Repeat with the long fixture only after the individual
alternate-background tests pass.

## Measurements

Record the following for baseline and alternate-pose runs:

- time to first live frame
- generated and delivered FPS
- average UNet time
- average VAE decode time
- average compose time
- dropped or repeated frames
- process RSS before session, during idle, and during live speech
- GPU memory before session and peak during live speech
- transition duration and any visible freeze
- mouth-to-audio synchronization
- visible mouth, jaw, cheek, mask-edge, or head-position drift

## Acceptance Criteria

The single-prepared-pose phase passes when:

- only one prepared avatar bundle exists for the test avatar
- peak GPU memory remains within normal run-to-run variance of the neutral-only
  baseline
- process RSS growth is bounded and returns after the session closes
- live FPS and time to first frame do not regress materially from baseline
- pose changes do not renegotiate or interrupt the WebRTC connection
- audio remains continuous and synchronized during live pose changes
- `speaking_direct` and `light_smile` have no obvious face-placement or mask-edge
  failures at normal playback size
- the final live-to-neutral transition is visually acceptable

For the first test, treat a throughput regression greater than 10 percent or a
repeatable time-to-first-frame increase greater than 250 ms as material.

## Failure Handling

If alternate composition becomes unstable during a session:

1. Stop using the alternate background.
2. Continue the same audio turn using the prepared neutral composition path.
3. Preserve the WebRTC sender and audio stream.
4. Record the active pose, frame indices, mapping data, and failure reason.

If visual drift is small but noticeable, test a lightweight correction before
considering full pose preparation:

- a fixed per-pose coordinate offset
- scale adjustment
- a slightly expanded or feathered neutral mask
- transition only at known compatible frame positions

If drift remains after this phase, the follow-up is full per-pose preparation
with a bounded runtime cache. That follow-up is documented below.

## Expected Outcome

The preferred result is a practical low-memory multi-pose avatar: one canonical
prepared neutral source, several streamed Kling pose MP4s, and live composition
onto whichever highly similar pose is active.

The experiment is intended to validate that visual shortcut before introducing
additional prepared cycles, cache pressure, GPU memory use, or S3 artifacts.

## Implemented WebRTC Controls

The session can now select idle playback and live composition poses
independently:

```text
POST /webrtc/sessions/{session_id}/idle-pose/{pose_id}
POST /webrtc/sessions/{session_id}/live-pose/{pose_id}
```

`POST /webrtc/sessions/create` also accepts `live_pose_id`. Omitting it keeps
the prepared `default` background path. Selecting another registered pose keeps
neutral prepared latents for inference while decoding that pose's MP4 as the
live composition background.

The session status response exposes `live_pose_id` and `live_pose_router`
diagnostics, including available poses, decoder activity, frame mapping, seeks,
and failures. A decode failure automatically falls back to the prepared neutral
background for the affected frames.

## Real Test Result: 2026-07-12

The first end-to-end test used the source image
`assets/landing_sample_recent_avatar_eng_60s_preview.jpg`, the generated
three-pose assets under `generated/segmind_pose_test/ab0eb7318f8f/`, and
`data/audio/yongen.wav`.

Only `neutral_resting.mp4` was prepared. `light_smile.mp4` and
`speaking_direct.mp4` were attached as idle/live pose MP4s without creating
additional prepared materials.

The real WebRTC schedule was:

```text
default -> speaking_direct -> light_smile -> speaking_direct -> default
```

All four live pose switches returned HTTP 200 while the stream was active. The
session recorded 103 frames, generated 80 live frames, had no dropped video
frames, and had no video stalls. Average GPU batch time was about 96 ms, with
about 30 ms UNet, 58 ms VAE, and 50 ms composition time. The neutral avatar
cache reported approximately 1.07 GB, confirming that no additional prepared
pose bundles were loaded.

Visual inspection:

- The neutral-only baseline was clean and lipsync behaved normally.
- The alternate live backgrounds produced visible mouth motion and preserved
  the broad avatar identity.
- `light_smile` and `speaking_direct` introduced noticeable face/mask drift at
  the cheeks and jaw because the neutral geometry was reused while the head
  angle changed.
- Pose changes were functional but visibly abrupt; the current route does not
  yet provide a visual-quality pass for production use.

Evidence files:

- `generated/segmind_pose_test/ab0eb7318f8f/webrtc_neutral_baseline.mp4`
- `generated/segmind_pose_test/ab0eb7318f8f/webrtc_pose_transition_test.mp4`
- `generated/segmind_pose_test/ab0eb7318f8f/webrtc_baseline_contact_sheet.jpg`
- `generated/segmind_pose_test/ab0eb7318f8f/webrtc_contact_sheet.jpg`
- `generated/segmind_pose_test/ab0eb7318f8f/face_transition_sheet.jpg`

Conclusion: the low-memory routing hypothesis is technically valid, but the
first visual experiment does not pass the smoothness criterion. The next
technical step should be lightweight per-pose face geometry or alignment
calibration, not additional prepared latent cycles.

## Queued Retry: 2026-07-12

The wall-clock pose switch test above did not validate all requested poses. The
shared scheduler composed most frames before the later API requests arrived, so
those later requests changed router state but not already queued video frames.

The retry uses a generated-frame pose queue configured before streaming. Each
segment is the full source MP4 duration at the session's 10 FPS:

```text
frames 0-48:    default / neutral_resting (4.9 seconds)
frames 49-97:   light_smile (4.9 seconds)
frames 98-146:  speaking_direct (4.9 seconds)
frames 147-195: light_smile (4.9 seconds)
after frame 195: hold light_smile
```

The queued WebRTC capture used a 22-second loop of the existing `yongen.wav`
fixture. It recorded 245 frames and generated 220 live frames with zero dropped
video frames and zero video stalls. The captured sequence visually follows the
requested order; the poses are not decoded or composited simultaneously.

Evidence:

- `generated/segmind_pose_test/ab0eb7318f8f/webrtc_queued_pose_transition_test.mp4`
- `generated/segmind_pose_test/ab0eb7318f8f/queued_face_transition_sheet.jpg`

The queued ordering is now valid. The `speaking_direct` segment still shows
some face/mask drift from reusing neutral geometry, which remains the next
visual-quality problem to solve.

## Full Prepared-Pose Validation: 2026-07-12

The queued retry established that the drift is caused by sharing neutral
prepared geometry, not by queue ordering. The test then prepared each tested
Kling clip as its own MuseTalk avatar bundle:

```text
base avatar / default: segmind_pose_test_ab0eb7318f8f
light_smile:           segmind_pose_test_ab0eb7318f8f_light_smile
speaking_direct:       segmind_pose_test_ab0eb7318f8f_speaking_direct
```

Each child bundle contains its own 294-frame cycle, masks, face coordinates,
blend plans, and latents. The base avatar persists the explicit pose map via:

```text
PUT /avatars/{avatar_id}/prepared-poses/{pose_id}?prepared_avatar_id={id}
```

At WebRTC stream setup, the scheduler loads the prepared pose bundles once.
For every queued generated frame it selects that pose's source-frame index,
conditioning latent, prepared background frame, mask, coordinates, and blend
plan. MP4-only poses retain the original low-memory background-decoder fallback.

### Validated Queue

```text
default -> light_smile -> speaking_direct -> light_smile -> hold light_smile
```

The final capture generated 220 live frames from the 22-second audio fixture,
with zero dropped video frames, zero video stalls, and no wall-clock pose
switches. The recorder now disables its legacy wall-clock schedule whenever a
generation-frame pose queue is supplied, preventing a test from clearing its
own queue.

### Visual Result

The full prepared-pose path removes the face/mask drift visible in the
neutral-only reuse run. Mid-segment samples and frames immediately before and
after all three queue boundaries show coherent cheeks, jaw, face scale, and
mouth placement. Expression changes remain, as expected from the source clips,
but there is no detached mouth or mask-edge artifact.

### Source Asset Finding

The remaining weak lip shapes are present even with the correct per-pose
prepared materials, so they are not caused by the WebRTC queue, pose routing,
or reuse of neutral masks. They come from the Kling-derived motion clips used
as source video. The next quality pass should therefore rework those source
assets: use a cleaner neutral mouth baseline, constrain idle motion to subtle
expression changes, and avoid source motion that already resembles speech or
has unstable lips. Re-prepare the affected pose bundle after each source-video
revision, then repeat this same queued visual test.

Evidence:

- `generated/segmind_pose_test/ab0eb7318f8f/webrtc_fully_prepared_pose_queue_test.mp4`
- `generated/segmind_pose_test/ab0eb7318f8f/fully_prepared_face_transition_sheet.jpg`
- `generated/segmind_pose_test/ab0eb7318f8f/fully_prepared_transition_boundaries.jpg`
- `generated/segmind_pose_test/ab0eb7318f8f/webrtc_fully_prepared_pose_queue_test.json`

Conclusion: separate prepared materials for each pose fix the visual alignment
failure for this test avatar. For a nine-session server, retain only the active
prepared pose and one or two upcoming queued poses per avatar in RAM; keep the
remaining prepared bundles on local disk/S3 behind an LRU cache.
