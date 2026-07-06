# Multiple Idle Video Variant Feasibility Analysis

Date: 2026-06-28

## Question

Can MuseTalk support multiple idle videos for the same avatar, with different
poses, movements, and head motion, instead of relying on one roughly 10 second
idle loop?

The proposed asset shape is:

- several idle videos for the same avatar
- durations such as 3s, 5s, 10s, 15s, 20s
- each video ends in the same position as the default idle video
- the shared ending position is intended to make transitions feel smooth

## Short Answer

Yes, this is possible, but only in a constrained version.

The current codebase can be extended to play multiple idle clips. That is mostly
a media-routing and metadata problem. The hard part is not playing the clips.
The hard part is guaranteeing smooth idle-to-live, live-to-idle, and
idle-variant-to-idle-variant transitions when the clips contain different body,
head, eye, and expression motion.

The important distinction:

- Playback-only idle variants are feasible.
- Smooth transitions are feasible only if the variants obey strict anchor rules.
- Arbitrary different poses and movements are not smooth by default, even if
  every clip ends on the same final pose.
- If live talking must start from any frame of any idle variant, then each
  variant needs frame-level mapping to the talking source, or needs to be
  prepared as a full MuseTalk source with its own latents, masks, coordinates,
  and frame cycle.

The practical recommendation is not to build the full general system now.
If this is still a product priority, build a limited "anchor-ended idle
variants" version first, with validation tooling and a strict interrupt policy.

## Prior Attempt To Read First

The recent lighter attempt is documented in:

- `docs/two_video_avatar_prep_handoff.md`
- related timing details are in `docs/hls_migration.md`
- current React Native behavior is summarized in `HLS_REACT_NATIVE_README.md`
  and `WEBRTC_REACT_NATIVE_README.md`

That attempt added support for:

- `video_file`: the talking/base video used for MuseTalk preparation
- `idle_video_file`: a separate idle loop used only while no audio is active

The key caveat from that work is the same caveat that applies here: when idle
and talking clips differ, continuity is only approximate. The current system can
show the separate idle video and can generate live talking from the talking
source, but it cannot guarantee that the body/head pose on the idle frame
matches the body/head pose used by the first generated live frame.

This is why the lighter version was not enough as a general smooth-transition
solution. It solved asset selection, not pose continuity.

## Current Architecture

MuseTalk does not generate full-body video from scratch. It prepares one source
video into reusable avatar materials:

- source frames
- face coordinates
- masks and mask coordinates
- VAE latents
- a frame cycle

In `scripts/api_avatar.py`, avatar preparation builds the live generation cycle
from the talking source:

```text
frame_list_cycle = frame_list + frame_list[::-1]
coord_list_cycle = coord_list + coord_list[::-1]
input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
```

Live generation uses this cycle for both latent selection and background frame
composition. The generated face crop is blended back into the selected talking
source frame.

The optional `idle_video_file` is different. It is stored as an idle playback
asset, but it does not create MuseTalk latents. It is not the source of live
composition frames.

This means the current two-video architecture is:

```text
idle mode:
  play idle_video.mp4

live mode:
  generate face crops from audio
  compose them into frames from input_video.mp4 cycle
```

That split is the root of the continuity issue.

## Existing Transition Support

The codebase already has useful transition machinery.

HLS:

- `scripts/hls_session_manager.py` creates a static idle HLS VOD manifest from
  one idle MP4.
- `api_server.py` computes a server-side live start offset using idle duration,
  expected live reveal delay, and avatar cycle length.
- `scripts/hls_gpu_scheduler.py` applies `start_offset_frames` to latent
  selection and composition.
- `templates/hls_player.py` uses separate idle/live video layers and a hold
  canvas to hide black frames during handoff.

WebRTC:

- `scripts/webrtc_tracks.py` has `SwitchableVideoStreamTrack`.
- It reads from one idle MP4 until live frames are ready.
- It can capture the currently displayed idle frame index.
- It maps that idle frame index to a MuseTalk cycle offset.
- It can hold the anchored idle frame while live frames prebuffer.

This makes single-video continuity much better because idle playback and live
generation are based on the same source. It does not fully solve separate idle
clips.

## Why "All Clips End In The Same Pose" Helps

Ending every idle variant at the same default pose is valuable.

It can make these transitions easier:

- idle variant -> default idle
- idle variant -> live talking from default/talking cycle frame 0
- live talking -> default idle, if the live output is also allowed to drain or
  return to a compatible frame

It also gives the system a stable anchor. A variant can be treated as a
decorative idle flourish that eventually returns to the canonical avatar state.

That is a workable design if live speech is only allowed to begin at the anchor,
or if the system is allowed to hold/freeze until the anchor is reached.

## Why Same Final Pose Is Not Enough

A single matching final frame does not guarantee a smooth transition.

The transition can still pop because of:

- different head velocity at the final frame
- different shoulder/body velocity
- different eye gaze
- different mouth closure
- different blink phase
- different expression tension
- different crop, scale, or face bounding box
- different lighting, compression, or color grade
- different playback FPS or frame cadence
- HLS decoder timing and segment boundary effects
- MuseTalk's ping-pong source cycle versus a forward-only idle MP4 loop

The most important problem is interruption. In a real conversation, audio may
arrive while a 20 second idle variant is halfway through. At that moment the
system has three choices:

1. Wait until the variant reaches the shared final pose.
2. Jump from the current variant pose into live talking.
3. Hide the jump with a freeze, hold frame, or crossfade.

Choice 1 preserves visual continuity but adds unpredictable latency. If the user
speaks at second 2 of a 20 second idle, waiting 18 seconds is unacceptable.

Choice 2 preserves latency but can visibly jump because live frames are composed
from the talking source, not from the idle variant.

Choice 3 can be acceptable for small pose differences, but it is cosmetic. It is
not a real pose-continuity guarantee.

## Feasible Designs

### Option A: Playback-Only Idle Variants

This is the smallest implementation.

Store several idle MP4s for one avatar and let the server/player choose among
them while the session is idle. The live generation path stays unchanged.

Required changes:

- extend avatar prep to accept multiple idle videos
- store `idle_variants.json`
- persist variants in S3
- expose `GET /avatars/{avatar_id}/video?role=idle&variant=...`
- generate or cache HLS idle manifests per variant
- make HLS and WebRTC sessions track the active idle variant
- expose active variant timing in status/debug output

This can work well for pre-call waiting, post-response breathing, or decorative
idle moments where the app controls when speech starts.

It is not enough for arbitrary live interruption. If a user or TTS response
arrives mid-variant, continuity is only approximate.

Verdict: possible, useful in limited cases, but not a complete smooth-transition
system.

### Option B: Anchor-Ended Variants With Interrupt Policy

This is the most practical version if we want to try the feature.

Each variant must have:

- a canonical entry anchor
- a canonical exit anchor
- an exit pose that matches the default idle/talking anchor
- a short return-to-anchor tail
- a maximum allowed time-to-anchor after interruption

Runtime policy:

- play variants only while idle
- when audio arrives, either finish the short return tail or hold the nearest
  compatible anchor frame
- start live generation from the matching talking cycle frame
- avoid long variants unless the system knows the avatar will remain idle long
  enough

This still has latency tradeoffs, but they are bounded if variants are designed
with short return tails.

Verdict: possible and probably the only version worth prototyping first.

### Option C: Dense Pose-Mapped Idle Variants

This version tries to start live speech from any frame of any idle variant.

Each idle frame needs a mapping to a compatible talking-source frame:

```text
idle_variant_id + idle_frame_index -> talking_cycle_frame_index
```

That mapping could be manually authored, generated from landmarks, or produced
by an offline validator. It still may fail when body pose differs, because
MuseTalk composes into the talking source background frame, not the idle
variant background frame.

For genuinely different poses and body motion, frame mapping alone is not
enough. The visual background will still jump unless the talking cycle contains
matching body/head frames.

Verdict: technically possible for carefully matched clips, but fragile.

### Option D: Full Per-Variant MuseTalk Preparation

This is the robust architecture for arbitrary poses.

Each idle/talking variant is prepared as a full MuseTalk source:

- frames
- coordinates
- masks
- latents
- compose plans
- cycle metadata

When live speech starts, the scheduler uses the active variant's prepared cycle
instead of the default talking cycle.

This would require:

- multiple prepared cycles per avatar
- scheduler jobs carrying `variant_id` or `cycle_id`
- avatar cache aware of multiple cycles
- S3 persistence for all cycles
- larger warmup/cache memory
- more prep time
- more validation

The storage and memory growth is roughly linear with total prepared video
duration. For example, variants of 3s + 5s + 10s + 15s + 20s at 20 FPS contain
about 1060 source frames before ping-pong cycling. A single 10s source at 20 FPS
contains about 200 source frames. Full preparation for those five variants is
therefore about 5.3x the source-frame volume before accounting for masks,
latents, compose plans, and S3 tarball size.

Verdict: possible, but much heavier. This should not be the first step unless
arbitrary current-pose live starts are a hard product requirement.

## HLS Impact

The current HLS design has one idle manifest per session:

```text
results/hls/{session_id}/index.m3u8
results/hls/{session_id}/segments/seg_*.m4s
```

Multiple idle variants require one of these designs:

1. Generate one idle manifest per variant and switch the idle video source.
2. Create a single server-authored idle playlist that can move between variants.
3. Replace static idle HLS with a continuous media pipeline.

Option 1 is simplest and fits the current architecture.

The player already has dual layers and hold-frame behavior. It could hide idle
variant switches the same way it hides live transitions, but it would need:

- variant-specific idle URLs
- decoded-frame gating before revealing a new variant
- active variant timing in `/status`
- server timing based on the active variant's duration and start time
- cache-busting so native HLS does not reuse stale manifests

The downside is that switching HLS sources is never free. The decoder may stall
or expose a blank frame unless the hold overlay remains visible until the new
variant has a decoded frame.

## WebRTC Impact

WebRTC is structurally easier because the server owns the video track.

The current `SwitchableVideoStreamTrack` wraps one `IdleVideoStreamTrack`.
Multiple variants would require a variant-aware idle provider:

```text
IdleVariantProvider
  active_variant_id
  active_track/container
  frame_count
  fps
  duration
  current_frame_index
  switch_to_variant(...)
  get_timing()
```

Switching should only happen while not live, or during a deliberate hold frame.
The timing object returned by `get_timing()` must include the active variant so
the live start offset can be computed correctly.

If we keep playback-only variants, WebRTC can hold the selected anchor frame
while live prebuffers. That helps hide the transition, but it still depends on
the active idle pose being compatible with the talking cycle.

## Asset Requirements

This feature depends more on asset discipline than code.

Every accepted idle variant should satisfy:

- same avatar identity, clothing, camera, framing, lighting, and resolution
- same FPS as the session target, or a clean conversion before upload
- no audio track, or an ignored/muted audio track
- no black frames
- no hard cuts inside the clip
- first and last anchor metadata
- exit anchor visually matches the canonical default idle anchor
- mouth closed or neutral at transition anchors
- eyes/gaze compatible at anchors
- final 200-500ms has low motion or a controlled ease into the anchor
- duration and frame count recorded by ffprobe

"Ends in the same position" should be enforced by tooling, not by eyeballing.

## Validator Needed Before Production

Before implementing runtime selection, build an offline validator.

Minimum validator checks:

- ffprobe duration, FPS, frame count, codec, resolution
- first frame and last frame extract
- compare exit anchor to default anchor with image difference
- face bounding box and landmark distance at anchor
- mouth-open score at anchor if available
- optical-flow magnitude over the final frames
- optional face crop contact sheet for visual review

The validator should produce a report per variant:

```text
variant_id
duration_seconds
fps
frame_count
exit_anchor_frame
default_anchor_frame
pixel_diff_score
landmark_diff_score
final_motion_score
pass/fail
failure_reason
```

Without this, the feature becomes subjective and hard to debug.

## Proposed Metadata

Example `idle_variants.json`:

```json
{
  "default_variant_id": "default",
  "canonical_anchor": {
    "variant_id": "default",
    "frame_index": 0,
    "talking_cycle_frame": 0
  },
  "variants": [
    {
      "id": "default",
      "path": "idle_variants/default.mp4",
      "duration_seconds": 10.0,
      "fps": 20.0,
      "frame_count": 200,
      "entry_anchor_frame": 0,
      "exit_anchor_frame": 199,
      "exit_talking_cycle_frame": 0,
      "interrupt_policy": "hold_to_anchor"
    },
    {
      "id": "idle_5s_head_turn",
      "path": "idle_variants/idle_5s_head_turn.mp4",
      "duration_seconds": 5.0,
      "fps": 20.0,
      "frame_count": 100,
      "entry_anchor_frame": 0,
      "exit_anchor_frame": 99,
      "exit_talking_cycle_frame": 0,
      "interrupt_policy": "finish_tail",
      "max_interrupt_tail_seconds": 0.5
    }
  ]
}
```

If full per-variant MuseTalk prep is later added, each variant also needs a
prepared cycle reference:

```json
{
  "id": "idle_5s_head_turn",
  "prepared_cycle_id": "idle_5s_head_turn",
  "latents_path": "variants/idle_5s_head_turn/latents.pt",
  "coords_path": "variants/idle_5s_head_turn/coords.pkl"
}
```

## Recommended Runtime Policy

Do not randomly start a long idle variant unless the app can tolerate finishing
or interrupting it.

Better policy:

- default idle loop is always safe
- after a live response ends, choose a short variant only if the system expects
  an idle gap
- prefer 3s and 5s variants for interactive conversation
- reserve 10s, 15s, and 20s variants for pre-call waiting or long passive idle
  states
- if audio arrives during a variant, use one of:
  - finish a short return-to-anchor tail
  - freeze the current frame and start live from the nearest safe anchor
  - abort to default idle with hold overlay, accepting a visible but masked
    correction

For low-latency conversation, the safest behavior is still the default idle
loop or very short variants.

## Implementation Plan If We Proceed

### Phase 0: Asset Spec And Validator

Create the idle variant asset specification and validation script. Do not change
runtime behavior yet.

Deliverables:

- `idle_variants.json` schema
- validator script
- example report
- pass/fail thresholds
- visual contact sheet output

### Phase 1: Playback-Only Variants

Add multiple idle MP4 storage and session selection.

Deliverables:

- `/avatars/prepare` supports multiple idle files or a zip/directory upload
- avatar metadata records variants
- S3 persistence includes variant videos and metadata
- HLS can create or link an idle manifest for the selected variant
- WebRTC can start with a selected variant
- status endpoints expose `idle_variant_id` and timing

Do not promise seamless interruption in this phase.

### Phase 2: Anchor-Ended Transitions

Add the constrained smooth path.

Deliverables:

- active variant timing is mapped to canonical anchors
- WebRTC can hold an anchor frame before live release
- HLS player can hold overlay while changing idle variants
- stream start uses the active variant's exit anchor when required
- long variants are blocked or downgraded in interactive mode

This is the first phase that can be product-tested for perceived smoothness.

### Phase 3: Full Per-Variant Prep, Only If Needed

If product requires live speech from arbitrary poses, prepare each variant as a
full MuseTalk source.

Deliverables:

- multiple prepared cycles per avatar
- scheduler jobs select cycle by `variant_id`
- cache sizing and eviction updated
- S3 tarball layout version bumped
- load tests repeated for memory, startup, and latency

This is the expensive version and should be justified by user-visible value.

## Risk Assessment

High-risk areas:

- unpredictable speech arrival during long variants
- visible body/head jumps because live composition uses another source
- HLS source switching stalls on native mobile players
- increased S3 and local cache size
- increased avatar prep time
- QA burden across every avatar/variant combination
- hard-to-debug subjective quality failures

Lower-risk areas:

- storing multiple MP4 idle assets
- selecting an idle variant at session creation
- exposing variant metadata in status endpoints
- WebRTC server-side switching while not live

## Final Recommendation

Do not implement the full "many idle videos with arbitrary poses" system as the
next production change.

It is technically possible, but the smoothness guarantee is not solved by simply
making every clip end in the same pose. That only gives us a safe anchor. It
does not solve mid-variant interruption or the fact that live MuseTalk frames
are composed from the prepared talking source.

If we want to test the idea, implement the constrained version:

1. Build the asset validator first.
2. Support multiple playback-only idle variants.
3. Allow variants only when idle.
4. Require every variant to return to the canonical anchor.
5. Use short variants for interactive conversation.
6. Treat long variants as pre-call or passive-idle assets only.

Only move to full per-variant MuseTalk preparation if users clearly notice and
value the added motion enough to justify the extra prep, storage, cache, and QA
cost.
