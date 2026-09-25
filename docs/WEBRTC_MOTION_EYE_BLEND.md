# Incoming-eye motion blend candidate

This optional profile addresses doubled eyelids during a transition from a held source frame into a moving blink. It refines only the eye region of the existing body bridge. It does not change source selection, audio timing, generated phonemes, source MP4s, prompts, or the separate MuseTalk beard-texture problem.

## Why this candidate exists

After correcting the planner to match the actual held frame, the Latina terminal return still softened/doubled eyelids. The held talking98 frame has closed eyes; incoming idle11→17 reopens them. Mixing the frozen closed eyelid colors into successive opening eyes creates an appearance that ordinary optical flow cannot reliably interpolate.

A CPU comparison tried higher-resolution full-frame flow and forward/backward consistency weighting. Full-frame refinement improved the specific eye artifact but cost about80–98ms per512×832 frame. A small region derived from actual source eye contours provides the useful detail without that full-frame cost.

`incoming_eye_blend` computes the normal global bridge, then warps a single incoming eye appearance toward the intermediate geometry. A feathered union of old/new six-point eye contours bounds this refinement. The incoming blink keeps its original frame sequence. There is no wait for open eyes and no loss of mandatory return coverage. Local mask construction is pixel-identical to the initial experiment across18 real bridge frames. Measured helper cost was16.39ms median /17.97ms p95 over54 CPU calls, including the global bridge; this excludes offline landmark extraction and is not a server throughput guarantee.

[CPU experiment and controls](/workspace/experiments/multipose_blend_quality_20260925/eyes/README.md) · [Helper parity/timing](/workspace/experiments/multipose_blend_quality_20260925/eyes/helper-parity.json).

## Runtime and integrity contract

An ordinary atlas has no `eye_blend` object and retains its original flow path. An explicitly enriched candidate contains:

```json
{
  "eye_blend": {
    "method": "incoming_roi_v1",
    "source_hashes": {"neutral_resting": "...", "speaking_direct": "...", "light_smile": "..."},
    "frames": {"neutral_resting": [], "speaking_direct": [], "light_smile": []}
  }
}
```

Each frames array must cover every original source frame. A row contains `left` and `right`, each six finite `[x,y]` pairs in source pixels. Closed, nearly collinear eyelids are valid. Startup checks source hashes, complete coverage, image dimensions and valid distinct contours. The complete profile contributes to the atlas routing digest. Missing or malformed geometry in an enabled profile fails explicitly; it does not silently substitute another character or profile.

`MotionBank.blend` dispatches the optional profile with exact source indices. The scheduler carries those indices through parallel composition and freezes the outgoing index along with the actual blend anchor across batches. Raw speech entry uses the current decoded idle frame on every tick. Interruption recovery uses the last emitted source phase and its selected consecutive idle frames, including wraparound. The existing bounded release, cancellation ownership and audio/generation timeline are retained.

An interruption during an unfinished bridge has an already warped anchor but only nominal original-source contour metadata. Its geometry is approximate. CPU coverage proves ownership/indexing, not perceptual accuracy of every partial blend; targeted live entry/body-bridge interruption recordings must remain part of validation.

## Reusable commands

Work from a prepared source-bound character package. MediaPipe is needed only by the offline collector. The runtime helper uses the already installed OpenCV/NumPy.

```bash
cd /workspace/MuseTalk
CUDA_VISIBLE_DEVICES= LIBGL_ALWAYS_SOFTWARE=1 \
/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/.venv/bin/python \
  scripts/measure_motion_eyes.py \
  --atlas /workspace/experiments/realtime_characters/latina_guided_20260925/motion-atlas.json \
  --output /workspace/experiments/NEW_EYE_PILOT/eye-measurements.json

/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/attach_motion_eye_profile.py \
  --atlas /workspace/experiments/realtime_characters/latina_guided_20260925/motion-atlas.json \
  --measurements /workspace/experiments/NEW_EYE_PILOT/eye-measurements.json \
  --output /workspace/experiments/NEW_EYE_PILOT/candidate/motion-atlas.json
```

Replace `NEW_EYE_PILOT` with a fresh directory. The collector verifies source hash, dimensions, FPS and frame count before publishing; the attachment tool rechecks the parent atlas hash and actual source files. It refuses to overwrite an atlas or registration sidecar, removes any inherited visual-review receipt, and publishes the new atlas as `candidate_requires_recorded_review`. Original source files, caches and approved package remain intact.

For an isolated pilot, set `WEBRTC_MOTION_ATLAS` to the new candidate atlas, leave `WEBRTC_MOTION_ATLAS_DIR` empty, and set `WEBRTC_MOTION_ALLOW_UNREVIEWED=1`. Use the existing native VP8 / TensorRT baseline launcher from the main plan. Registering both original and candidate banks with the same three source hashes in one registry is deliberately rejected as ambiguous. This separate atlas is not automatically a complete browser-lab character package.

Pass the candidate explicitly as `--atlas` to `scripts/test_webrtc_motion_transitions.py`, while using the original package's prepared `--pose-set`. Reuse the original source clips and audio. Run the default short/long/interruption cases, then separate `--case entry-interruption` and `--case edge-cases` output directories. Use `scripts/review_motion_evidence.py` with the same candidate atlas; a different routing digest must fail verification. Rollback is selecting the original atlas/registry again; no source or cache regeneration is needed.

## Tasks and evidence

- [x] Reproduce the terminal/interrupted eye artifact on exact raw source pairs.
- [x] Compare full-frame, consistency-weighted and local incoming-eye alternatives.
- [x] Implement source-bound offline contours, candidate attachment and runtime dispatch.
- [x] Keep ordinary banks on the byte-identical baseline path; validate enabled metadata before use.
- [x] Test frozen source indices across batches, actual entry ticks, cancellation and return wraparound.
- [x] Complete five live candidate recordings and automated timing checks; retain inspected stills and open artifacts.
- [ ] Exercise this profile on Japanese and Indian characters before generalizing the result.
- [ ] Check eye texture steps, ROI seams, partial-transition interruption and normal-speed quality.
- [ ] Validate glasses, hair occlusion, other face shapes and poses before using it for arbitrary user portraits.

The candidate passed285 Python regressions and five actual Latina recordings (1,702 saved frames at exact20Hz;11returns in0.3988–0.4508s). Existing15 Node browser-driver tests are unchanged from the preceding checkpoint and were not rerun for this eye-only change. The five live cases include partial-entry/body-bridge interruption and a following reply, silence, threshold length and looping. These prove exercised behavior and timing, not invisible transitions. Live stills show a cleaner eyelid contour, but remaining nose/upper-lip doubling must not be labelled solved by an eye-only change.

Current experiment root: `/workspace/experiments/multipose_blend_quality_20260925`. `eye-runtime-inputs.json` captures runtime hashes before the live candidate launch. `eye-candidate/motion-atlas.json` binds all723 Latina contour rows. `run-eye-pilot.sh` and `run-eye-edges.sh` contain exact receiver commands. Final results and open issues are recorded in that folder's README and the committed validation snapshot. No candidate is marked visually reviewed.

## Separate beard-texture finding

A passive capture retained238 exact256×256 MuseTalk outputs. Independent reconstruction matched every original background, mask, decoded-face and final-composite hash. Thus the Indian beard softening exists before encoding. A spatial mouth envelope retained more outer beard while keeping every current lip-core pixel exact across75 evaluated frames, but moustache/core softness and jaw-motion tradeoffs remain. This is an offline diagnosis, not an enabled mask change. [Exact evidence and silent comparison](/workspace/experiments/multipose_blend_quality_20260925/texture/exact-inputs/README.md).

The body bridge still blends already composed frames, so it can mix old/new phonemes. Eye-only refinement preserves that existing mouth behavior. Any future separation of body warping and current mouth composition must apply the same intermediate geometry to the mouth/mask; passing a warped background into the existing composer alone is insufficient.
