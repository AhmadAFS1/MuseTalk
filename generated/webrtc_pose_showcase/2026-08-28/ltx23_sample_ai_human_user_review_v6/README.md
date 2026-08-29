# LTX 2.3 user-review v6 WebRTC session

This directory contains a real end-to-end MuseTalk WebRTC receiver session for the v6 six-pose bank, not a source-video concatenation.

## Review files

- `sample_ai_human_ltx23_user_review_six_v6_labeled_webrtc.mp4` — labeled 480×832, 24 fps H.264/AAC review encode.
- `sample_ai_human_ltx23_user_review_six_v6_webrtc_capture.mp4` — untouched receiver recording.
- `labeled_webrtc_contact_sheet.jpg` — representative idle and live frames.
- `labels.ass` — telemetry-aligned label source.
- `run_metadata.json` — concise machine-readable session evidence.
- `webrtc_smoke.log` — complete test output.
- `superseded_short_neutral/` — the earlier technically passing capture that switched away from neutral too quickly for useful breathing review.

## Session result

- Result: passed.
- Session ID: `2SuDIhBsKoEivvZUbCPtDw`.
- The receiver held neutral resting for 11.1 seconds before queuing the other poses, so the revised breathing is visible for a complete cycle.
- All six independently prepared/warmed v6 caches were observed in order: neutral, active listening, speaking direct, promoted nod, slow empathy, light smile, and neutral again.
- The same session then accepted 60 seconds of audio and played 1,200 live MuseTalk frames using light smile, speaking direct, and neutral resting.
- Receiver totals: 2,881 video frames and 6,104 audio frames.
- Final state: neutral resting with no active stream.
- First-live audio/video RTP delta: 5.0 ms, below the 41.7 ms one-frame allowance at 24 fps.
- Timestamp validation passed; the one video source timestamp discontinuity is paired with the declared 21-frame phase correction recorded by the worker.

The v6 pose set remains a draft visual-review candidate. Passing the WebRTC and boundary tests does not by itself mark the motion production-approved.
