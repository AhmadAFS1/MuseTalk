# LTX 2.3 user-review v7 WebRTC session

This directory contains a real end-to-end MuseTalk WebRTC receiver session for the v7 six-pose bank, not a source-video concatenation.

## Review files

- `sample_ai_human_ltx23_user_review_six_v7_labeled_webrtc.mp4` — labeled 480×832, 24 fps H.264/AAC review encode.
- `sample_ai_human_ltx23_user_review_six_v7_webrtc_capture.mp4` — untouched receiver recording.
- `labeled_webrtc_contact_sheet.jpg` — representative idle and live frames.
- `labels.ass` — telemetry-aligned label source.
- `run_metadata.json` — concise machine-readable session evidence.
- `webrtc_smoke.log` — complete test output.
- `superseded_off_lens_speaking_remaster/` — first technically passing v7 session, superseded because its speaking keyframe kept an off-lens glance too long.

## Session result

- Result: passed.
- Session ID: `ksG4JM637Dh-SmWSHivsew`.
- All six independently prepared/warmed v7 caches were observed in order after a full 11.1-second neutral opener.
- The same session accepted 60 seconds of audio and played 1,200 live MuseTalk frames.
- Receiver totals: 2,968 video frames and 6,289 audio frames.
- Final state: neutral resting with no active stream.
- First-live A/V RTP delta: 10 ms, below the 41.7 ms one-frame allowance at 24 fps.
- The labeled review fully decoded at 480×832, 24 fps, with AAC audio.

The v7 pose set remains a draft visual-review candidate. Passing the WebRTC and boundary tests does not by itself mark the motion production-approved.
