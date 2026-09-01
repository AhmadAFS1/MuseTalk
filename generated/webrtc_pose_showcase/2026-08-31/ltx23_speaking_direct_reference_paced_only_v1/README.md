# V15 reference-paced direct-only WebRTC review

This session tests V15, a direct-speaking LTX 2.3 source informed by real direct-to-camera human footage. The chosen reference is Mixkit's “Therapist in his office talking to the camera”: a fixed-camera take with frontal attention, stable shoulders, calm intervals, and occasional compact speed changes.

Review `sample_ai_human_ltx23_speaking_direct_reference_paced_only_v1_webrtc_labeled.mp4` first. It labels the pre-roll, 60-second live MuseTalk interval, and post-roll. The untouched receiver recording is `sample_ai_human_ltx23_speaking_direct_reference_paced_only_v1_webrtc_capture.mp4`.

V15 retains V14's direct gaze, one uninterrupted source segment, fixed shoulders and torso, and exact canonical boundaries. It changes motion timing rather than amplitude. Measured p90 center speed is 1.76× V14, while p95 center travel is 32% smaller. The 24-sample gaze audit contains no off-axis samples.

The source achieved the planned speed target. Its displacement fell below the planned middle band, making V15 a quicker-but-tighter candidate. The first quick action also rendered as a compound emphasis rather than two perfectly isolated peaks. These are disclosed limitations, and playback review remains the acceptance test.

All six pose IDs required by the WebRTC protocol map to the same V15 source and prepared avatar cache. Reaction intent is disabled, so no alternate idle, nod, empathy, or smile footage appears.

The WebRTC run passed with 1,200 generated frames, zero dropped frames, zero queue underruns, zero strict audio/video stalls, zero timestamp anomalies, and no phase correction. First-live A/V RTP skew was 8.3 ms against a 41.7 ms allowance. The untouched and labeled recordings fully decode. V15 remains a debug candidate and does not replace V14 or the approved v8 assets.
