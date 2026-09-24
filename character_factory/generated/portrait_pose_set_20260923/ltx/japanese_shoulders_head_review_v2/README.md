# Japanese three-pose LTX review set — cropped guide

This review set uses the same Japanese portrait and native LTX 2.3 low-VRAM
graph as the September 23 test, with the corrected `center_crop` guide. The
[before/after guide comparison](../japanese_guide_fit_before_after.jpg)
shows why the old shoulder edges appeared stretched: the old image repeated
about 22 pixel columns at each side. All three selected clips use one new
guide hash (`181034ff3a68660506ba6bcf77e6c2de545c73b70eac264872d57927de88dee2`).

| Pose | Video | Contact sheet | Prompt/seed source | Review observation |
| --- | --- | --- | --- | --- |
| Idle | [idle.mp4](idle.mp4) | [idle-contact.jpg](idle-contact.jpg) | `crop_level_head_v2`, seed 197 | Clean shoulder edge, subtle breathing and blinks; 8.0 px vertical eye travel. |
| Talking | [talking.mp4](talking.mp4) | [talking-contact.jpg](talking-contact.jpg) | `crop_expression_anchored_v5`, seed 197 | Clear lip motion, steady gaze; 14.5 px vertical eye travel (9.5% of eye distance) versus the liked Indian native clip's 8.8%. |
| Smiling | [smiling.mp4](smiling.mp4) | [smiling-contact.jpg](smiling-contact.jpg) | `crop_smile_seed191_v8`, seed 191 | Closed-lip smile, no pronounced sideways tilt; a brief upward head lift remains (18.5 px vertical eye travel). |

The exact selected text is in
[`japanese_selected_native_three_pose_v2.json`](../../../../config/prompt_packs/japanese_selected_native_three_pose_v2.json).
The [validation](validation.json) matches every MP4's hash to its original
render manifest and checks 512×832, 24 fps, 241 frames, no audio, and literal
decoded first/last equality **within** each clip. The
[motion audit](motion-audit.json) uses FaceMesh on all 241 decoded frames per
clip. Generation and decode graphs are in [`graphs/`](graphs/). The complete
iteration and rejection record is in
[`JAPANESE_SHOULDER_HEAD_RERUN_20260924.md`](../../../../JAPANESE_SHOULDER_HEAD_RERUN_20260924.md).

This is a review set, not a switch-safe MuseTalk pose bank. Endpoint hashes
still differ across poses; shared-anchor certification and human approval at
normal playback speed remain necessary. The smile's brief lift was reduced
from other cropped closed-lip candidates but was not eliminated by prompt
wording. No MuseTalk override or WebRTC run was performed for this rerender.
