# LTX 2.3 variable-pose MuseTalk WebRTC review

This run uses six distinct LTX 2.3 motion plates generated from the same portrait:

1. `neutral_resting` — breathing, uneven blinks, and natural head sway
2. `active_listening` — acknowledgement nods and engaged head sway
3. `speaking_direct` — passionate head emphasis, facial energy, and shoulder movement
4. `nod_agree` — one clear agreement nod with shoulder follow-through
5. `empathetic_head_tilt` — a broad, gentle sideways sway
6. `light_smile` — a subtle closed-lip smile with breathing and a blink

## Review files

- `sample_ai_human_ltx23_variable_six_v2_labeled_idle_review.mp4` — 30.6-second labeled pass through the complete idle bank
- `sample_ai_human_ltx23_variable_six_v2_labeled_webrtc.mp4` — 49.7-second labeled real WebRTC capture, including live TTS and MuseTalk lip sync
- `sample_ai_human_ltx23_variable_six_v2_webrtc_capture.mp4` — untouched WebRTC recorder output
- `labeled_webrtc_contact_sheet.jpg` — quick visual index
- `labels.ass` — telemetry-aligned label source

## Validation

- Each source clip is 480×832, 24 fps, 121 frames (5.041667 seconds), with no audio.
- The first six and final six decoded frames in every certified clip are the same canonical frame.
- Every certified clip has the same decoded boundary RGB SHA-256:
  `47b05c6bdd63466e13381dc6cf21545e827bea0bc668c5798cbf7c69f7076b33`.
- All 30 ordered cross-pose concatenation/decode checks passed.
- All six avatar caches were prepared and warmed before recording.
- The WebRTC session completed with `ok: true`; its first live audio/video RTP timestamps were aligned within 20 ms (41.67 ms allowance).
- Both labeled MP4s passed complete FFmpeg decode checks.

The pose manifest remains marked `switch_safe: false` because this is a visual evaluation bank, not a production-approved asset set.
