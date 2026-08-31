# Speaking-direct 10° isolated WebRTC review

This debug session uses only the LTX 2.3 10° `speaking_direct` motion plate. All six pose IDs required by the WebRTC pose protocol map to the same source video and the same prepared MuseTalk avatar cache. Reaction intent is disabled, so the receiver never substitutes the approved idle, listening, nod, empathy, or smile footage.

Review `sample_ai_human_ltx23_speaking_direct_10deg_only_v1_webrtc_labeled.mp4` first. The labels mark the pre-roll, the 60-second live MuseTalk interval, and the post-roll. The unmodified receiver recording is `sample_ai_human_ltx23_speaking_direct_10deg_only_v1_webrtc_capture.mp4`.

The run passed: 1,200 generated video frames were played with zero dropped frames, zero audio/video stalls, valid receiver timestamps, and 3.3 ms first-live A/V skew. This remains a debug candidate and does not modify or promote the v8 pose bank.
