# V14 + V15 direct-speaking WebRTC comparison

Review `sample_ai_human_ltx23_v14_v15_speaking_comparison_v1_webrtc_labeled.mp4` first. It contains one continuous MuseTalk/WebRTC call using the same 60-second speech track while alternating between the two direct-speaking candidates.

The live sequence is V14 from 0.0-24.0 seconds, V15 from 24.0-48.1 seconds, then V14 from 48.1-60.0 seconds. Those live-relative intervals appear at 3.75-27.75, 27.75-51.85, and 51.85-63.75 in the receiver recording because it includes pre-roll.

The V14 to V15 handoff used a four-generation-frame crossfade at the requested 24.0-second cue. The V15 to V14 handoff snapped to the nearest certified boundary at 48.1 seconds, only 0.1 seconds after its requested cue. The complete three-segment rendered trace matched the compiled plan.

The test passed with 1,200 generated frames, zero dropped frames, zero queue underruns, zero strict audio stalls, and zero strict video stalls. The received WebRTC recording timestamp audit also passed. The untouched receiver capture is retained beside the labeled review video.
