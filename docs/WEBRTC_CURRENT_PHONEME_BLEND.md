# Current phoneme during multipose transitions

2026-09-25. Experimental, source-bound `current_similarity_v1` profile. The
original banks and approved LTX source videos remain unchanged. This profile
addresses pose changes during generated speech; it does not certify invisible
transitions or replace the separate entry and post-audio recovery policies.

## Problem and implementation

The previous compositor freezes an already lip-synced outgoing frame and blends
it with every incoming frame. Its old phoneme therefore persists across the
six-frame transition. Exact captured Indian inputs demonstrate a wide current
mouth becoming almost closed in the old blend.

The optional API layer return preserves the original composed pixels and carries
the exact prepared raw body plus the effective parser alpha. The alpha covers
only the generated face's pasted area, including its soft edges. Read-only views
share prepared cache storage without changing cached flags; normalized alternate
backgrounds own their raw bytes. These layers travel in the existing ordered
composition batch, never in a separate queue.

On a source change, the scheduler holds the preceding raw source and alpha for
the entire transition. Dense optical flow aligns the outer body. Across both
face supports and eye contours, a similarity transform derived from four stable
eye corners aligns the head while preserving normalized current mouth geometry.
The incoming raw image, composed image and alpha use exactly the same map. A
feathered eye region selects the current incoming eye appearance, retaining a
single blink phase.

The current composed image already contains its parser mask. After remapping,
the final expression is:

```text
output = warped_current_composed
       + (1 - warped_current_alpha) * (blended_raw_body - warped_incoming_raw)
```

Clip once to the display range. Applying alpha to the composed image again would
suppress soft-edge generated detail. Full parser-core pixels are exactly the
warped current composition at every transition step. The helper has no outgoing
composed-face argument and therefore cannot mix in a previous generated phoneme.
No frames or audio samples are skipped or retimed.

This is an active-speech operation: progress zero still carries current speech
at outgoing body geometry. It must not replace the generic idle-release helper.
The existing incoming-eye profile remains in use for entry and recovery. Current
speech uses the new shared face/eye similarity mapping explicitly described here.

## Configuration and reproduction

1. Keep the original source package and prepared API caches.
2. Measure eyes with `scripts/measure_motion_eyes.py` in the existing MediaPipe
   environment, then attach them using `scripts/attach_motion_eye_profile.py`.
   Neither MediaPipe nor face detection runs in the live helper.
3. Create a separate candidate:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python \
  scripts/attach_motion_current_profile.py \
  --atlas /path/to/eye-candidate/motion-atlas.json \
  --output /path/to/new-current-candidate/motion-atlas.json
```

The publisher checks source bytes and refuses an existing output/registration.
It removes previous visual approval. The manifest profile binds all three source
hashes and requires complete measured eye geometry and equal frame dimensions.
The profile changes the routing identity. Configure only the candidate registry;
duplicate banks for the same sources are intentionally rejected.

4. Run the existing server with the native VP8 backend and
   `WEBRTC_MOTION_ATLAS=/path/to/new-current-candidate/motion-atlas.json`,
   `WEBRTC_MOTION_ATLAS_DIR=` and `WEBRTC_MOTION_ALLOW_UNREVIEWED=1`.
5. Record using `scripts/test_webrtc_motion_transitions.py --atlas ...` and verify
   with `scripts/review_motion_evidence.py`. Inspect the received transitions,
   including initial phonemes and partially completed bridges. Passing timing
   checks does not grant visual acceptance.

The actual candidate and pilot commands are under
`/workspace/experiments/multipose_current_phoneme_20260925/`.
`runtime-inputs.json` binds the exact code and atlas loaded by the pilot.

## Validation and tasks

- [x] Reconstruct captured generated faces and exact prepared composites.
- [x] Compare 24 actual transition frames at the scheduler's cosine weights.
- [x] Preserve current mouth core, soft alpha and normalized articulation shape.
- [x] Keep the same incoming geometry for body, face, mask and eye appearance.
- [x] Verify production helper is byte-identical to the corrected CPU prototype.
- [x] Preserve the default API and original bank behavior.
- [x] Reject missing indices/layers/fade counts and incompatible dimensions.
- [x] Test out-of-order completion, frozen anchors, cancellation during blending
      and callbacks, failed-job isolation, and final layer cleanup.
- [x] Run 285 existing regressions and 40 additional API/helper/runtime tests.
- [x] Review three fresh received videos and their timing reports; retain visual failures.
- [ ] Repeat the candidate on Japanese and Indian live sessions.
- [ ] Verify actual browser and mobile-client behavior.
- [ ] Obtain normal-speed perceptual acceptance before promoting any bank.

The CPU proof uses 48 exact reconstructed inputs for 24 blend frames. Its
single-thread median is 32.18 ms and p95 33.41 ms, versus approximately 10.23 ms
for the previous full-frame blend. These are helper timings, not service or
multi-user capacity measurements. Temporary maps exist only during transitions;
prepared per-frame raw/mask payloads are shared views. Alternate backgrounds and
an unbounded compose backlog still need capacity evaluation for broader service
use.

The captured Indian generated face itself has softer beard texture. This change
preserves that generated appearance and does not repair the texture. The separate
mask experiment remains offline. Arbitrary post-audio release, actual browser
playback, and human normal-speed acceptance remain distinct requirements.

## Received pilot result

Three actual Latina recordings passed the source/routing, saved-frame cadence
and return-timing audits: 1,416 frames at exact 50 ms intervals, eight returns in
0.3992–0.4498 seconds, and zero reported strict video stalls. Long speech covers
all four scheduled source changes. The interrupted run includes the following
short reply. Edge cases cover threshold speech, silence, cancellation during
entry/body motion, and a longer loop. This is a sequential one-GPU pilot.
The observed maximum pending compose count was two, not a capacity guarantee.

**Visual acceptance failed.** The new long recording's smile-entry frames
154–156 still show doubled nostrils and a vertical upper-lip/philtrum seam.
Current articulation is preserved, and the inspected terminal body transition
is cleaner, but most old/new live comparisons use different source pairs.
Only the controlled Indian CPU reconstruction establishes an identical-input
comparison. The result must remain experimental. See
[the received review](/workspace/experiments/multipose_current_phoneme_20260925/live-review/README.md)
and [the live gallery](/workspace/experiments/multipose_current_phoneme_20260925/latina-live/review.html).

Next work should isolate nose/upper-lip appearance using captured exact Latina
raw/composed/alpha inputs for the failing pair. Preserve the current-mouth core
and shared incoming geometry while checking whether the central face needs a
single incoming appearance or a better local spatial map. Do not compensate by
fading current phonemes again. Re-test partial-transition recovery afterward.

The actual browser launch remains pending after automatic approval review
rejected downloading/executing an external browser on the nearly full root
filesystem. Dependency provisioning was subsequently approved; browser execution
was not. No browser or mobile playback is claimed by these receiver recordings.

Additional edge finding: the one-second `silence.wav` input is exactly zero
(24,000 samples at 24 kHz, measured peak and RMS both zero), yet the generated
idle-body mouth opens at receiver frames 141–147. This occurs outside the new
transition helper and needs a separate silence policy that preserves actual
speech. Partial-body interruption also includes a three-frame build hold; it
is recorded and must not be described as continuous motion. See the edge review
and `silence-input-check.json` in the current experiment directory.
