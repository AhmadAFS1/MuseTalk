# V14 gaze-locked subtle-motion direct-only WebRTC review

This session tests V14, which keeps V13's straight camera gaze and adds restrained talking-head motion. The source permits only two slow, unequal forward/chin micro-adjustments across twelve seconds. It does not permit a side glance, yaw, roll, lateral sway, shoulder response, or torso motion.

Review `sample_ai_human_ltx23_speaking_direct_gaze_locked_subtle_motion_only_v1_webrtc_labeled.mp4` first. It labels the pre-roll, 60-second live MuseTalk interval, and post-roll. The untouched receiver recording is `sample_ai_human_ltx23_speaking_direct_gaze_locked_subtle_motion_only_v1_webrtc_capture.mp4`.

All six pose IDs required by the WebRTC protocol map to the same V14 source and prepared avatar cache. Reaction intent is disabled, so no alternate idle, nod, empathy, or smile footage appears.

The source gaze audit passed with zero off-axis samples. Its 95th-percentile central head travel is 11.78 px: 71.3% more than V13 but 54.7% less than V12. The yaw-like proxy remains 1.47 px and eye-line roll remains 0.70 degrees.

The WebRTC run passed with 1,200 generated frames, zero dropped frames, zero queue underruns, and zero strict audio/video stalls. First-live A/V RTP skew was 15 ms against a 41.7 ms allowance. One declared one-frame video phase correction occurred at live handoff; receiver timestamp validation passed. The untouched and labeled recordings fully decode. This remains a debug candidate pending user review and does not modify the approved v8 assets.
