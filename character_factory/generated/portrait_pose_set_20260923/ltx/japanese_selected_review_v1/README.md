# Japanese three-pose LTX 2.3 prompt review

These are the best **observed** candidates from the September 23–24 Japanese
portrait prompt test. The MP4s and contact sheets here are symlinks to the
original, unmodified render outputs. They are for playback review, not a
certified MuseTalk pose bank.

| Pose | Video | Contact sheet | Source candidate |
|---|---|---|---|
| Idle | [idle.mp4](idle.mp4) | [idle-contact.jpg](idle-contact.jpg) | locked-head candidate v2, seed 197 |
| Talking | [talking.mp4](talking.mp4) | [talking-contact.jpg](talking-contact.jpg) | liked Indian native talking prompt, gender words changed, seed 191 |
| Smiling | [smiling.mp4](smiling.mp4) | [smiling-contact.jpg](smiling-contact.jpg) | surviving V4 small-smile wording, seed 193 |

All three clips are silent, 512×832, 24 fps, and 241 frames (10.04 seconds).
Independent decode checks found exact first/last RGB equality within each
clip. The three endpoint RGB hashes differ, so transitioning between them
requires the existing common-anchor certification step. LTX talking motion
has no supplied TTS audio; MuseTalk lip-sync override was not tested here.

`validation.json` records file hashes, frame/stream checks, decoded endpoint
hashes, and full-frame face landmark measurements. The exact prompts and
provenance are in
`/workspace/MuseTalk/character_factory/JAPANESE_NATIVE_THREE_POSE_PROMPTS_V1.md`;
the selected runnable prompt pack is
`/workspace/MuseTalk/character_factory/config/prompt_packs/japanese_selected_native_three_pose_v1.json`.

Candidate results still need human review at normal playback speed. In
particular, the talking clip moves its head vertically more than the accepted
Indian body plate, and the selected smile moves vertically more than the
Indian V8 smile. No WebRTC session or production cache update was made.
