# Speaking-direct 3°-target half-again isolated WebRTC review

This session tests a fresh LTX 2.3 direct-speaking take whose requested movement distance is approximately 50 percent smaller than the v10 half-distance candidate. It reuses the same generation seed pair, five prompt segments, four-to-five speaking events, and ordinary conversational timing. The prompt asks for ordinary head changes near 1–2° and an absolute 3° cone, but LTX does not expose a geometric head-angle constraint or measurement; `3°` is a prompt target, not a proven physical maximum.

Review `sample_ai_human_ltx23_speaking_direct_3deg_half_again_only_v1_webrtc_labeled.mp4` first. It labels the pre-roll, 60-second live MuseTalk interval, and post-roll. The untouched receiver capture is `sample_ai_human_ltx23_speaking_direct_3deg_half_again_only_v1_webrtc_capture.mp4`.

All six pose IDs required by the WebRTC protocol map to the same source video and one prepared avatar cache. Reaction intent is disabled, so no alternate idle or reaction footage appears.

The run passed with 1,200 generated video frames, zero dropped frames, zero strict audio/video stalls, and no receiver timestamp anomalies. First-live A/V skew was 6.7 ms, within the 41.7 ms allowance. This remains a debug candidate and does not modify the approved v8 assets.
