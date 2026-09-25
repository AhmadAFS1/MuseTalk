# Smooth transitions for the approved Japanese three-pose bank

Status: analysis and proposed implementation, 2026-09-25. No runtime changes,
new video renders, or WebRTC tests were performed for this analysis.

The approved idle, talking, and smiling clips remain the source masters. Their
prompt approval establishes appearance quality; it does not certify switching
from every interior frame.

## Recommendation

Separate speech playback from the body-motion source. Start speech on the
currently visible prepared source, and change body clips only through reviewed
transitions. Stop audio at its real endpoint or on interruption; let body motion
settle through a short reviewed exit while the mouth closes naturally.

For the first pilot, use the idle source for replies shorter than three seconds.
For longer replies, enter the talking source only if a good entry and a bounded
exit route exist. Three seconds is the user's proposed policy threshold, not
a property of the video or a measured perceptual threshold.

Matching source endpoints permits endpoint switching. It cannot make an
arbitrary mid-clip cut smooth. A three-second talking loop only reduces the
worst wait to nearly three seconds; it does not solve arbitrary interruption.

## What the current repository actually does

| Behavior | Implementation | Consequence |
| --- | --- | --- |
| One persistent video sender | scripts/webrtc_tracks.py, SwitchableVideoStreamTrack | A transition can stay inside the existing track without SDP renegotiation. |
| Start speech at the idle phase | scripts/hls_gpu_scheduler.py calls capture_idle_sync_timing and align_first_queued_pose | Matching phase is meaningful for the same source motion. It does not align two independent LTX motions. |
| Prefer a nearby full-loop boundary | scripts/webrtc_pose_router.py, _compile_pose_plan_locked | Safe endpoints may be many seconds from an audio cue. |
| Otherwise force a pose change at the requested time | Same compiler, requested_time_crossfade | Default fallback requests four generated frames of crossfade. |
| Blend a held outgoing frame over incoming frames | scripts/hls_gpu_scheduler.py, _apply_webrtc_pose_crossfade | Can conceal a small residual difference but can also show two displaced face outlines. It does not match motion velocity. |
| Preload idle frame zero and activate it at audio completion | api_server.py around stage_completion_idle_video; scripts/webrtc_tracks.py, end_live and _activate_completion_idle_video | The completion path does not match the last visible talking frame to idle. It can cut halfway through the talking motion. |
| Use prepared materials per pose | scripts/hls_gpu_scheduler.py, latent gathering and composition | The current runtime can select matching pose-specific latents and backgrounds. Reusing idle face geometry on another pose is unnecessary for this design. |
| Restrict speech plans and manifests | scripts/pose_protocol.py | Speech allowlist excludes neutral_resting; manifests require six logical IDs. A physical three-video bank needs explicit internal source mapping or a versioned protocol update. |

The four-frame crossfade is approximately 267 ms at 15 fps generation, or
167 ms at 24 fps generation. It is not a fixed millisecond policy.

The July architecture document accurately describes transport synchronization
work, but its successful timing checks do not certify visual continuity for
the new Japanese clips.

## Evidence from these exact approved videos

Source:
[/workspace/experiments/japanese_ltx_fixed_distance_20260925/framing-validation.json](/workspace/experiments/japanese_ltx_fixed_distance_20260925/framing-validation.json).

- All six first/last delivery frames have one identical decoded RGB hash.
- At source time 2.0 seconds, the talking eye midpoint is 15.28 pixels above
  its reference, while idle is 0.47 pixels below it: about 15.74 pixels apart.
- Maximum eye-height deviation is 17.58 pixels for talking, versus 5.74 for idle.
- First/last adjacent-frame differences are about 2.9–3.0 RGB levels, while
  idle's interior 95th percentile is 1.70. Equal endpoints alone do not make
  their neighboring motion continuous.
- Talking has visible mouth articulation in its source. Playing the untouched
  remainder after speech stops would allow silent mouthing.

These are existing 2D detector/pixel diagnostics. They do not establish a
literal change in identity or measure every transition artifact. The practical
problem is mismatched pose, expression, texture, and movement between separately
generated videos of the same character.

## Offline preparation: build routes between motion sequences

1. Keep the approved MP4s and hashes unchanged. Store transition derivatives
   separately and identify the source hashes in their manifest.
2. Prepare each physical video for MuseTalk with its own face crops, masks,
   coordinates, and latent cycle. Validate the actual crop used by each cache.
3. Measure stable facial landmarks, face scale, head rotation, shoulders, hair
   outline, background appearance, blink state, and motion over short windows.
   Compare corresponding outgoing and incoming windows, not one still frame.
4. Find candidate idle-to-talking and talking-to-idle connections throughout
   the clips. The destination may be an interior idle frame. There is no need
   to jump to idle frame zero on every return.
5. Reject candidates with large geometry, lighting, identity-texture, blink, or
   motion-direction mismatch. A shared endpoint is a useful candidate, not the
   only possible candidate.
6. Start by testing direct joins between close matches. For remaining small
   mismatch, test a short motion-aligned bridge offline. A tentative search
   range is 150–300 ms; this is a pilot parameter, not a validated result.
   Optical flow or landmark-guided alignment may help, but can distort hair,
   jaw edges, and occlusions. Discard visibly warped or ghosted bridges.
7. Include lower-face/mouth checks. Exclude rapidly changing lips from the
   initial body matching score, then validate the actual rendered mouth and jaw.
   A good upper-face match does not certify the final lip-synced transition.
8. Save only reviewed connections: source pose/frame, destination pose/frame,
   transition length, confidence, cache mapping, motion direction, and
   supported speech/silence mode.
9. Calculate exit coverage for EVERY playable source phase. A proposed first
   target is an idle return within 500 ms total, including any wait and bridge.
   This target must be tested. Report the worst exit delay, not just an average.
10. If a region cannot reach a reviewed exit within the budget, do not enable
    that region in an interruptible talking path. Either create a dedicated
    return segment for it or keep that reply on the shared idle source.

A small collection of precomputed approved routes is sufficient for a pilot;
we do not need a real-time generative video model. The approach is inspired by
[Video Textures](https://www.microsoft.com/en-us/research/publication/video-textures/)
and [Motion Graphs](https://graphics.cs.wisc.edu/Papers/2002/KGP02/), which
synthesize controllable motion from compatible recorded sequences and
transitions. Applying that idea to these LTX/MuseTalk assets is a proposed
adaptation, not an existing proven feature of this repository.

## Runtime policy

Maintain separate speech and motion states on one continuous media timeline.

| Event | Audio / mouth | Body motion |
| --- | --- | --- |
| Short reply, under 3 s | Normal MuseTalk lipsync | Continue idle at its current phase |
| Longer reply begins | Start with ordinary A/V prebuffer; do not add delay to await body switching | Initially continue current idle source |
| Reviewed entry becomes available | Keep audio and lipsync continuous | Enter talking through the approved route |
| Planned sentence approaches its end | Continue lipsync until the final phoneme | Prefer an approved return to idle before audio ends |
| Normal audio completion | Natural mouth release to resting state | Continue the current idle phase; do not reset to frame zero |
| User interrupts mid-sentence | Cancel unsent speech and stale inference promptly | Begin the reviewed exit for the last emitted body phase |
| Smile requested | Mouth follows audio if speaking | Use a reviewed smile entry/exit; otherwise skip or defer the smile |

For known complete TTS, plan the exit ahead of time. For example, a 4.6-second
reply could return its body to idle at 4.3 seconds while the last 0.3 seconds
of speech is still lip-synced over idle. Those times are illustrative, not
measured routes in the present clips.

For an unexpected stop, the already transmitted receiver buffer cannot be
retracted. Cancel unsent speech without waiting for the body, and use the most
recent emitted pose/source-frame state to plan subsequent output. Keep the
persistent audio/video clocks monotonic.

### Mouth behavior during an interrupted return

Do not simply resume the raw talking source after muting its audio. It already
contains speech-shaped lips. The return needs a validated lower-face treatment
that closes naturally while preserving the original face size and head motion.

Candidate: an offline prepared silent companion for the talking/return frames,
using the same geometry and source phase, with a short mouth-release sequence
from the live result. A silence-conditioned MuseTalk pass is only a candidate:
silent audio does not guarantee a naturally closed mouth, and earlier tests
raised a concern about shrinking lips. Review the lip width, jaw shape, mask
edges, and transition from the last live phoneme explicitly.

If that companion fails review, do not promise immediate silent returns from
all talking phases. Keep the shared idle source fallback until usable returns
are available. Endpoint padding or a long face dissolve does not solve this.

## Concrete implementation changes

- Add a versioned transition manifest separate from the semantic pose plan.
  Store source hashes, time bases, validated edges, and maximum exit coverage.
- Replace the implicit equal-phase assumption across different physical clips
  in LivePoseVideoRouter with transition lookup. Equal phase remains a useful
  shortcut only for truly shared source motion.
- Carry pose/render key, original source frame, cache frame, direction, and turn
  ownership through inference, composition, and video queue entries. The queue
  currently stores generation ownership and an image, but not the source pose
  and phase needed to recover from the displayed point.
- Track the last frame actually emitted by SwitchableVideoStreamTrack. The
  generation cursor can be ahead because of prebuffering. Do not derive an
  interruption route from that future cursor.
- Split speech completion from motion completion. end_live currently clears
  the live frame and activates the staged idle frame. Preserve the outgoing
  pose state and hand it to a session-owned return controller first.
- Keep request ownership protections. Release the audio turn without a stale
  completion callback overwriting a newer body route. A new utterance during
  a return continues lipsync on the current prepared route, then replans.
- Keep source/cache indices explicit across 24 fps source, generation fps,
  and WebRTC output fps. APIAvatar currently creates forward-plus-reversed
  cycles; verify direction and cycle mapping rather than assuming one modulo
  index represents all three clocks. Avoid accidentally reversing talking,
  breathing, or blinks.
- Preserve pose-specific latent/background pairing. An in-speech bridge needs
  prepared bridge frames or another explicitly validated compose path.
- Keep the six-ID public protocol compatible via explicit internal source
  mapping, or introduce capability-negotiated support for a smaller bank.
  Do not silently change the existing speech allowlist or alias a smile to
  an unintended speech pose.

## First experiment and acceptance gates

Before changing the live server, create a small offline transition review from
the existing accepted videos:

- compare current immediate idle return, endpoint-only return, and best matched
  candidate/bridge at several interior points, including 2, 4, 6, and 8 seconds;
- measure exit coverage across all 241 frames, in both directions;
- inspect source motion first, then repeat with actual MuseTalk lipsync and
  the candidate silent return;
- use the same-source idle/lipsync path as the continuity baseline;
- play captures at normal speed, checking head/shoulder jumps, double eyes,
  ghosted hair, flash-like exposure changes, shrinking lips, and silent talking.

Only enable live multipose for routes that pass that review. Then exercise
short replies, just-over-threshold replies, normal ends, mid-phoneme
interruptions, interruptions during a bridge, back-to-back turns, long
silences, and queue underruns. Log pose/source phase at every emitted frame,
audio stop timing, exit latency, and stale-frame rejection. Correct timestamps
and equal boundary hashes are necessary checks, not proof of an invisible cut.

The implementation can guarantee that only reviewed routes are selected.
Whether these existing ten-second videos contain enough good routes remains
unmeasured. If they do not, use one source for idle and speech while adding
purpose-built exit clips; keep the approved trio as the visual reference.
