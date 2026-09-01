# LTX 2.3 full-shoulder MuseTalk WebRTC proof

The final proof files for `sample_ai_human_ltx23_facetime_wide_production_v1` are:

- `sample_ai_human_ltx23_full_shoulders_six_pose_webrtc_capture_v2.mp4`: every logical pose followed by one 12-second direct-speaking turn.
- `sample_ai_human_ltx23_wide_v14_v15_two_turn_webrtc_capture_v3.mp4`: two direct-speaking turns in one persistent WebRTC session, proving deterministic V14-to-V15 rotation.

Both captures passed the receiver timestamp validator and fully decode as 480×832 H.264 video with AAC audio. `run_metadata.json` records the session IDs, pose trace, rotation trace, and validation result.

Earlier captures in this directory are retained as additional evidence; none were overwritten or deleted.
