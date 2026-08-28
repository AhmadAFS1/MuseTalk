# LTX 2.3 balanced v4 WebRTC session

Created on 2026-08-26 from the six certified `balanced-conversational-v4` LTX 2.3 pose clips.

## Review files

- `sample_ai_human_ltx23_balanced_six_v4_labeled_webrtc.mp4` — labeled 24 fps review encode of the WebRTC receiver output.
- `sample_ai_human_ltx23_balanced_six_v4_webrtc_capture.mp4` — untouched WebRTC receiver recording with received audio.
- `labeled_webrtc_contact_sheet.jpg` — representative frames from all six idle poses and the three live-speaking phases.
- `labels.ass` — telemetry-aligned label source used for the review encode.
- `run_metadata.json` — concise machine-readable session results.

## Session flow

The receiver observed all six idle clips in this order before live MuseTalk speech was submitted:

1. Neutral resting
2. Active listening
3. Speaking direct
4. Nod / agreement
5. Empathetic head tilt
6. Light smile
7. Return to neutral resting

The live section then used light smile, speaking direct, and neutral resting while MuseTalk generated the speaking stream. The labels distinguish idle-clip playback from live MuseTalk output.

## Validation result

- Result: passed
- Session ID: `HfXd-SqKArSeKOYC2HzXLg`
- Pose set: `sample_ai_human_ltx23_balanced_six_v4`
- Elapsed test time: 49.29 seconds
- Receiver video frames: 1,154
- Receiver audio frames: 2,438
- Live MuseTalk frames played: 250
- Final pose: neutral resting
- Final stream state: inactive
- First-live audio/video RTP timestamp delta: 6.7 ms, below the one-frame 24 fps limit of 41.7 ms
- Audio timestamp anomalies: zero
- Recording timestamp validation: passed
- Every one of the six avatar caches was prepared and warmed independently before the session.

The labeled video was fully decoded after encoding. The source recording remains available separately so the review labels do not replace or obscure the original evidence.

## Review caveat

This pose bank is still a draft review candidate (`test_only: true`, `switch_safe: false`). The retained nod is the best of five generated candidates, but its eyes close during the downward part of the nod. The WebRTC label calls this out explicitly. Do not mark this bank production-ready until that behavior is accepted or replaced.
