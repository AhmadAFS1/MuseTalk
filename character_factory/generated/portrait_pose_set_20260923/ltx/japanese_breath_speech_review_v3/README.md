# Japanese avatar: shallow idle and longer talking, review set v3

These are the best measured review candidates from the September 24 LTX 2.3
rerolls. They use the same Japanese source portrait and `center_crop` guide as
the [frozen v2 set](../japanese_shoulders_head_review_v2/README.md). The exact
selected text, seed, native frame count, and idle repeat count are in
[`japanese_selected_native_three_pose_v3.json`](../../../../config/prompt_packs/japanese_selected_native_three_pose_v3.json).

| Pose | Watch | Contact sheet | Native source | Key observation |
| --- | --- | --- | --- | --- |
| Idle | [idle.mp4](idle.mp4) | [idle-contact.jpg](idle-contact.jpg) | [81-frame idle-cycle.mp4](idle-cycle.mp4), seed 197, repeated 3× | Three small breath cycles and three blinks; 5.2 px maximum shoulder-midpoint travel versus 14.1 px in v2. |
| Talking | [talking.mp4](talking.mp4) | [talking-contact.jpg](talking-contact.jpg) | Native 241-frame v5, seed 191 | Varied lip motion into second 8; last central lip gap above 3 px at 8.71 s versus 3.92 s in v2. Shoulder travel fell from 17.2 to 6.3 px. |
| Smiling | [smiling.mp4](smiling.mp4) | [smiling-contact.jpg](smiling-contact.jpg) | Unchanged v2, seed 191 | Same previously liked restrained smile; no new LTX render. |

All three delivered files are silent 512×832 H.264, 24 fps, 241 frames
(10.04 seconds), with identical decoded first and last frames **inside each
clip**. The [validation](validation.json) records their hashes and stream
metadata; the [motion audit](motion-audit.json) measures every decoded frame.
The [idle assembly report](idle-assembly.json) confirms identical decoded
frames at the four loop boundaries. No fade, crossfade, MuseTalk, or SoulX
was applied. The idle is explicitly assembled from a short native LTX cycle,
not generated in one ten-second pass.

The talking clip no longer goes silent halfway, but its final 1.25 seconds
are quiet before the forced matching endpoint. Its prompt asks for speech
through the last moment; the actual video does not satisfy that literally.
The idle blinks at approximately even 3.3-second intervals because the short
cycle repeats. Review both clips at normal playback speed for the cadence,
blink rhythm, shoulder edges, and any flicker before approving them.

To generate a new three-pose review set with this exact behavior, run
`character_factory/scripts/generate_three_pose_videos.py` with the source
portrait, `--prompt-pack character_factory/config/prompt_packs/japanese_selected_native_three_pose_v3.json`,
and `--guide-fit center_crop`. Its `repeat_cycles` support builds the
241-frame idle from the 81-frame LTX generation. The separate
[`repeat_short_idle_cycle.py`](../../../../scripts/repeat_short_idle_cycle.py)
can reproduce the selected idle from `idle-cycle.mp4` without rerunning LTX.
The [controlled comparison notes](../../../../JAPANESE_SHALLOW_BREATH_CONTINUOUS_TALK_20260924.md)
record the rejected candidates and measurements.

This is a **review set**, not a switch-safe MuseTalk pose bank. The three
endpoint hashes differ across poses, so shared-anchor certification,
normal-speed human approval, and runtime cache creation are still required.
