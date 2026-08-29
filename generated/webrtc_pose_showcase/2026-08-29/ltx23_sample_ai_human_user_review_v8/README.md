# LTX 2.3 user-review v8 WebRTC session

This directory contains a real end-to-end MuseTalk WebRTC receiver session for the v8 six-ID compatibility pose set. It is not a source-video concatenation.

## Review files

- `sample_ai_human_ltx23_user_review_six_v8_labeled_webrtc.mp4` — labeled 480×832, 24 fps H.264/AAC review encode.
- `sample_ai_human_ltx23_user_review_six_v8_webrtc_capture.mp4` — untouched receiver recording.
- `labeled_webrtc_contact_sheet.jpg` — representative idle and live frames.
- `labels.ass` — telemetry-aligned label source.
- `run_metadata.json` — concise machine-readable session evidence.
- `webrtc_smoke.log` — complete test output.

## Session result

- Result: passed.
- Session ID: `kk3X5hBjWIkjtYmIwypo2Q`.
- Idle and active listening were both observed through the same approved v6 active-listening cache and motion bytes.
- All six compatibility pose IDs were observed in order.
- The same session accepted 60 seconds of audio and played 1,200 live MuseTalk frames.
- Receiver totals: 2,914 video frames and 6,173 audio frames.
- Final state: merged idle motion under the `neutral_resting` compatibility ID with no active stream.
- First-live A/V RTP delta: exactly 0 ms against a 41.7 ms one-frame allowance at 24 fps.
- The labeled review fully decoded at 480×832, 24 fps, with AAC audio.

The v8 pose set remains a draft visual-review candidate. Passing the WebRTC and boundary tests does not by itself mark the motion production-approved.
