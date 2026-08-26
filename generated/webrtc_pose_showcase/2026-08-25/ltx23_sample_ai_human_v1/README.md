# LTX 2.3 six-pose WebRTC review

This directory contains a receiver-side recording of a real local WebRTC session through the MuseTalk API. It is not a simple offline concatenation of the source clips.

## Files

- `sample_ai_human_ltx23_six_v1_webrtc_review.mp4` — 45.5-second review cut. It keeps the full six-pose idle circuit and complete talking segment, while removing the silent model-generation wait.
- `sample_ai_human_ltx23_six_v1_webrtc_capture.mp4` — untouched 60.5-second receiver recording, including the generation wait.
- `session_contact_sheet.png` — representative frames across idle and speaking playback.
- `idle_transition_contact_sheet.png` — before/after frames around the five idle-pose handoffs.

## Observed pose sequence

| Receiver time | Pose |
| ---: | --- |
| 0.383 s | `active_listening` |
| 5.235 s | `speaking_direct` |
| 10.524 s | `nod_agree` |
| 14.557 s | `empathetic_head_tilt` |
| 19.642 s | `light_smile` |
| 24.769 s | `neutral_resting` |

The assistant response then used `light_smile` -> `speaking_direct` -> `neutral_resting` while MuseTalk generated lip motion from `data/audio/ai-assistant.mpga`.

## Validation notes

- All six avatar caches were prepared and remained resident during the session.
- The untouched capture fully decodes and contains H.264 video at 480x832 with AAC stereo audio at 48 kHz.
- The review cut is H.264/AAC at 480x832 and 24 fps, with `faststart` enabled for browser playback.
- The five idle handoffs keep camera, crop, identity, lighting, and background aligned. The `nod_agree` handoff has the largest measured frame-to-frame change, but visual inspection shows no hard scene or identity jump.
- The media session completed. The test harness subsequently reported a strict timestamp-proof mismatch: the first received TTS audio packet was 20 ms later than the declared marker (one 48 kHz audio frame). This did not interrupt recording, pose switching, audio playback, or video decoding.

