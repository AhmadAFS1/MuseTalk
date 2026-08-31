# Speaking-direct half-distance isolated WebRTC review

This session tests a fresh LTX 2.3 direct-speaking take whose movement distance is approximately 50 percent smaller than the preceding 10° take. It keeps the same generation seed pair, five prompt segments, four-to-five speaking events, and ordinary conversational timing. Only motion amplitude was intentionally reduced: typical head travel is 2.5–4° with a hard 5° limit.

Review `sample_ai_human_ltx23_speaking_direct_half_distance_only_v1_webrtc_labeled.mp4` first. It labels the pre-roll, 60-second live MuseTalk interval, and post-roll. The untouched receiver capture is `sample_ai_human_ltx23_speaking_direct_half_distance_only_v1_webrtc_capture.mp4`.

All six pose IDs required by the WebRTC protocol map to the same source video and one prepared avatar cache. Reaction intent is disabled, so no alternate idle or reaction motion appears.

The run passed with 1,200 generated video frames, zero dropped frames, and zero strict audio/video stalls. A one-frame video RTP phase correction was declared at the live handoff; first-live A/V skew was 16.7 ms, within the 41.7 ms allowance, and timestamp validation passed. This remains a debug candidate and does not modify the approved v8 assets.
