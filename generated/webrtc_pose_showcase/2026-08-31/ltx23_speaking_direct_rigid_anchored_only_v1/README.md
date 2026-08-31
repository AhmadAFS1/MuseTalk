# Speaking-direct rigid/anchored isolated WebRTC review

This session tests the V12 LTX 2.3 `speaking_direct` source after measurement showed that the preceding “3°” prompt produced almost the same dominant head travel as V10. V12 does not use degree wording. It removes the forced five-event trajectory, mixed-axis turns, chin emphasis, forward-neck motion, shoulder follow-through, anti-stillness instructions, and reused seeds. It instead uses three steady-state prompt segments, a fresh seed pair, and an anchored head/neck/shoulder posture.

Review `sample_ai_human_ltx23_speaking_direct_rigid_anchored_only_v1_webrtc_labeled.mp4` first. It labels the pre-roll, 60-second live MuseTalk interval, and post-roll. The untouched receiver recording is `sample_ai_human_ltx23_speaking_direct_rigid_anchored_only_v1_webrtc_capture.mp4`.

All six pose IDs required by the WebRTC protocol map to the same V12 source and one prepared avatar cache. Reaction intent is disabled, so no alternate idle, nod, empathy, or smile footage appears.

The run passed with 1,200 generated video frames, zero dropped frames, zero strict audio/video stalls, and 10 ms first-live A/V RTP skew against a 41.7 ms allowance. A declared one-frame video phase correction occurred at live handoff; receiver timestamp validation passed. Both recordings fully decode. This remains a debug candidate pending visual review and does not modify the approved v8 assets.
