# Speaking-direct gaze-locked isolated WebRTC review

This session tests V13, a substantially more rigid LTX 2.3 direct-speaking source that removes V12's off-axis glance. The prompt contains no eye-focus variation, breathing cue, brow event, posture event, or degree wording. One uninterrupted segment holds both pupils at the lens and keeps the face, head, neck, shoulders, and torso registered; only symmetrical blinking is allowed.

Review `sample_ai_human_ltx23_speaking_direct_gaze_locked_only_v1_webrtc_labeled.mp4` first. It labels the pre-roll, 60-second live MuseTalk interval, and post-roll. The untouched receiver recording is `sample_ai_human_ltx23_speaking_direct_gaze_locked_only_v1_webrtc_capture.mp4`.

All six pose IDs required by the WebRTC protocol map to the same V13 source and one prepared avatar cache. Reaction intent is disabled, so no alternate idle, nod, empathy, or smile footage appears.

The run passed with 1,200 generated video frames, zero dropped frames, zero strict audio/video stalls, and 11.7 ms first-live A/V RTP skew against a 41.7 ms allowance. One declared one-frame video phase correction occurred at live handoff; receiver timestamp validation passed. Both recordings fully decode. This remains a debug candidate pending user review and does not modify the approved v8 assets.
