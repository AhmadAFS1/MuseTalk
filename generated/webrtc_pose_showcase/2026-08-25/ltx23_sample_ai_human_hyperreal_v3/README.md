# LTX 2.3 hyperreal conversational v3 WebRTC review

This run replaces the exaggerated v2 motion bank with six low-amplitude conversational behaviors:

1. `neutral_resting` — faint breathing, brief ordinary blinks, and no intentional head motion
2. `active_listening` — one shallow acknowledgement with an attentive expression
3. `speaking_direct` — small conversational emphasis while the shoulders remain planted
4. `nod_agree` — one compact agreement nod with no repetition
5. `empathetic_head_tilt` — a slight held inclination and softened expression
6. `light_smile` — a restrained closed-lip smile with stable posture

## Review files

- `sample_ai_human_ltx23_hyperreal_six_v3_labeled_idle_review.mp4` — labeled 30.9-second pass through the complete idle bank
- `sample_ai_human_ltx23_hyperreal_six_v3_labeled_webrtc.mp4` — labeled 50.3-second real WebRTC capture with live TTS and MuseTalk lip sync
- `sample_ai_human_ltx23_hyperreal_six_v3_webrtc_capture.mp4` — untouched WebRTC recorder output
- `labeled_webrtc_contact_sheet.jpg` — visual index of all six idle poses and the live phase
- `labels.ass` — telemetry-aligned label source

## Review and corrections

- Every accepted source was reviewed as a multi-frame sequence; endpoint certification alone was not treated as acceptance.
- Two neutral generations and two nod generations were rejected for prolonged eye closure.
- The final neutral source retains the accepted restrained LTX body motion, with its sustained eye closure temporally compressed to a six-frame blink. Removed time was redistributed over the quiet open-eyed recovery.
- The final nod source retains the accepted compact LTX nod, with its nod/eye-closure passage compressed to twelve frames. Removed time was redistributed over the open-eyed recovery.
- Rejected and original-timing takes are preserved under the LTX bank's `rejected/` directory and are not installed as MuseTalk avatars.

## Validation

- Each accepted clip is 480×832, 24 fps, and 121 frames (5.041667 seconds), without audio.
- The first six and final six decoded frames in every certified clip are the same canonical frame.
- Every clip has decoded boundary RGB SHA-256 `47b05c6bdd63466e13381dc6cf21545e827bea0bc668c5798cbf7c69f7076b33`.
- All 30 ordered cross-pose concatenation/decode checks passed.
- All six v3 MuseTalk avatar caches were prepared and warmed before recording.
- The WebRTC session completed with `ok: true`.
- First-live audio/video RTP timestamps were aligned within 11.7 ms (41.67 ms allowance).
- Both labeled MP4s passed complete FFmpeg decode checks.

The pose set remains `draft`, `test_only`, and `switch_safe: false` until visual approval.
