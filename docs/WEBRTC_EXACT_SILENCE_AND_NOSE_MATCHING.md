# Exact silence and nasal alignment checkpoint — 2026-09-25

The silent-upload mouth artifact is addressed in the WebRTC compositor. The broader multipose quality goal remains open. A controlled smiling-entry comparison improves the doubled nostrils, but matching alone cannot cover every interruption phase.

## Implemented behavior

`scripts/webrtc_exact_silence.py` decodes the original uploaded audio before downmixing, resampling or integer normalization. A nonempty, single audio stream is classified as exact silence only if every decoded sample in every channel is digital zero. One nonzero sample, including a floating-point subnormal, retains the ordinary speech path. PCM8 uses its unsigned midpoint. Multiple streams, empty media, unknown formats and decode failures retain ordinary validation. This is not an RMS threshold or a per-frame speech detector.

The timeline preserves the original path, duration and samples for a whole-zero upload, including when edge trimming is enabled. The server passes the resulting flag internally to the scheduler; it is not a client-controlled switch. With a motion bank, the scheduler selects only the neutral source and emits its raw frames instead of the generated mouth. It preserves the original inference, audio, frame-count, entry, completion and cancellation machinery. The job exposes the flag in scheduler telemetry. HLS behavior remains unchanged.

The decoder helper supports a cancellation event, but the existing timeline preparation does not pass it; the existing shielded setup/cancellation behavior remains authoritative. This change does not optimize away GPU inference. Without a motion bank it retains the existing source selection: a source that itself talks is not guaranteed to have a closed mouth. Nonzero recordings, including near-silent audio and recordings containing internal pauses, still use normal lipsync.

## Tests and actual received evidence

**347 Python tests pass:** 287 existing/updated regressions in 12.446 seconds and 60 focused current-phoneme/silence regressions in 0.850 seconds. The focused set includes 20 new exact-silence tests; two new endpoint tests are in the larger suite. Coverage includes original-media precision, opposite-sign stereo channels, ambiguous streams, malformed/empty audio, actual timeline preparation, internal endpoint forwarding, idle-only routing, raw composition, cancellation and unchanged frame counts. An earlier run's obsolete zero-audio rejection assertion and missing fixture field failed; its log is retained separately.

The final actual server loaded the working implementation directly, without the diagnostic capture hooks. One received edge-case session covers threshold speech, one second of exact-zero audio, interruption during body motion, a following short reply and a longer looping reply. The strict verifier passes: **882 saved video frames at exact 50 ms spacing**, five completed returns in **0.3986–0.4407 seconds**, and no receiver timestamp anomalies. Runtime and atlas hashes matched the frozen inputs after recording. The test server was stopped and its GPU released.

- [Received video](/workspace/experiments/multipose_nose_silence_20260925/silence-live/edge-cases.mp4)
- [Review page](/workspace/experiments/multipose_nose_silence_20260925/silence-live/review.html)
- [Strict verification](/workspace/experiments/multipose_nose_silence_20260925/silence-live/verification.json)
- [Independent silence inspection](/workspace/experiments/multipose_nose_silence_20260925/silence-review/README.md)

The silent turn preserves all 20 generated/played frames and 50 audio packets (1.000 second), uses only neutral source frames, and is the only one of five turns classified as exact silence. Inspection of the received frames shows the closed mouth closely follows the exact prepared raw source; the old silent test had an open generated mouth. Lossy video/audio encoding prevents received pixel/sample equality, and the old/new recordings use different source phases. The following reply starts with a fresh generation after interruption.

These are local aiortc receiver recordings, not browser/mobile validation or normal-speed perceptual acceptance. The launcher used `WEBRTC_TRIM_EDGE_SILENCE=0`; the trim-enabled whole-zero behavior is covered by CPU tests, not a separate live run.

## Controlled nasal alignment diagnosis

The earlier current-phoneme recording doubled the nostrils when talking frame 52 entered smiling frame 63. Exact generated faces were captured using a frozen copy of commit `3a9ccc2`; all 12 selected prepared composites and alpha masks were reconstructed exactly. After the runtime's eye-corner similarity map, the nose-tip mismatch starts at 9.34 pixels and grows to 11.77 pixels. Soft face-mask edges retain competing raw nostrils.

Two CPU alternatives were rejected. Selecting a single incoming nose appearance removes the double image but jumps its position. A local nose correction preserves the mouth pixels but pinches/elongates the nose and philtrum; its minimum spatial Jacobian is 0.179. Translating the lower face preserves mouth shape but visibly moves it. Neither experiment changed production code.

A diagnostic atlas changes only talking 52 → smiling 63 to talking 52 → smiling 54. Both received comparisons start from idle frame 49, use the same frozen runtime and source files, and have identical generated face hashes for the four captured frames before the switch. In the inspected received sequence, target 54 has one coherent nostril pair, preserves current articulation, and uses a single closing/reopening eye appearance. It improves this specific entry. Later source phases differ as a consequence of the new entry, so later transitions are not identical-input comparisons.

- [Received controlled comparison](/workspace/experiments/multipose_nose_silence_20260925/phase54-review/README.md)
- [Original target 63](/workspace/experiments/multipose_nose_silence_20260925/fixed63-live/long-talking-smiling.mp4)
- [Diagnostic target 54](/workspace/experiments/multipose_nose_silence_20260925/fixed54-live/long-talking-smiling.mp4)
- [Rejected local controls](/workspace/experiments/multipose_nose_silence_20260925/cpu/README.md)

Target 54 fails the original eye-aperture gate because it starts in a blink. Its low nose-tip error does not mean the whole nose aligns: one nostril still differs by 7.90 pixels. This manual edge is diagnostic only and was not installed in the final silence test or a character package.

## Why a matcher alone is insufficient

An offline audit measures all 723 source frames using explicit anatomical nose landmarks and stable eye corners. It explores removing the aperture gate only for banks with the incoming-eye profile, while retaining the other geometry, appearance, dynamics and idle-mouth constraints. It evaluates nasal mismatch across the entire six-frame blend window, rather than only its first frame. An illustrative score selects smiling frame 52, with 54 fourth; it is not a production scoring constant.

At an illustrative weighted nasal error budget of 0.02 outer-eye spans (about 3 pixels), only **197/241 talking phases** and **171/241 smiling phases** have a qualifying idle target. Even the best available targets for the hardest phases differ by 14.30 and 12.53 anchor-equivalent pixels. Loosening the threshold until coverage reaches 100% would conceal this limitation. See [measurements and coverage](/workspace/experiments/multipose_nose_silence_20260925/matching/README.md).

## TASKS and next execution order

- [x] Classify original-media digital zero without suppressing quiet phonemes.
- [x] Keep exact-zero requests on raw neutral frames with unchanged timing.
- [x] Run the 347-test regression set and an actual edge-case receiver session.
- [x] Capture and reconstruct the failing nose transition with exact inputs.
- [x] Compare target 63 and 54 with the same initial phase and unchanged sources.
- [x] Measure all source phases and retain the insufficient return coverage.
- [ ] Develop a source-bound, nose-aware matcher and test a separate candidate; do not silently relax quality gates or ship the manual edge.
- [ ] Provide a face alignment bridge for the phases no target can match, while preserving current articulation and natural nose/mouth geometry.
- [ ] Re-test returns from the actual emitted partially blended face, not just the nominal source frame; inspect the existing build hold.
- [ ] Address beard texture using the captured Indian evidence, without weakening phoneme preservation.
- [ ] Repeat the accepted candidate on Japanese and Indian identities and interruptions across difficult phases.
- [ ] Run actual browser and mobile tests, then obtain normal-speed visual acceptance before marking a bank reviewed.

Start further work from the exact captured inputs. Evaluate the whole nasal base and philtrum, current mouth geometry, eyelids, shoulders and camera distance together. Preserve mandatory return timing and the beginning of every utterance. Keep rejected candidates and comparisons separate from accepted source masters. No new LTX render was run for this checkpoint.

## Reproduction and resources

Evidence root: `/workspace/experiments/multipose_nose_silence_20260925/`. Its README records the evidence split and commands. `silence-runtime-inputs.json` binds the actual final server; the controlled nose runs use the separately frozen baseline and capture hooks. `capture-inputs.json` is an early attempt and must not be treated as the final silence server manifest.

The root filesystem had only about 66 MB free after the recording. New temporary work uses `/dev/shm/musetalk-nose-capture-tmp` and `PYTHONDONTWRITEBYTECODE=1`. Provenance journals record relocation of a regenerable TensorRT timing cache and 2,878 CPython bytecode files to task-owned shared memory, with verified unchanged Python sources. No model, source clip or user evidence was deleted. The shared-memory cache copies are temporary, not required evidence. UV offline pruning found no unused entries.

Real browser execution remains pending: automatic approval review rejected downloading/extracting and executing an external browser outside its sandbox on the nearly full root filesystem. A later dependency-only action was approved; it did not approve browser execution. The mobile repository is absent. These limitations do not prevent local compositor work, but must not be presented as completed client validation.
