# Japanese portrait: shoulder-edge and head-motion rerun

This is the September 24, 2026 review of the Japanese portrait's three native
LTX 2.3 poses. It follows the selected September 23 review set. The source is
the same `generated/portrait_pose_set_20260923/japanese_woman.png`; the model,
512×832 frame, 24 fps, 241 frames, two exact endpoint guides, and delivery
encoding remain the same. The new outputs are review clips, not a certified
MuseTalk pose bank.

## Why the shoulders looked stretched

The earlier guide preparation fit the 941×1672 portrait by height, then filled
the remaining 44 horizontal pixels by repeating the outermost portrait
columns, about 22 pixels on each side. In the lower 250 rows, the mean
horizontal pixel difference inside either edge strip is exactly zero. This
created an unnatural static stripe adjoining the shoulders before LTX saw the
image; prompt wording could not repair that input. The
[before/after guide](generated/portrait_pose_set_20260923/ltx/japanese_guide_fit_before_after.jpg)
shows the cause directly.

The defect survives generation: in the old idle and talking videos, the
average horizontal difference inside either lower edge strip is only
0.09–0.19 pixel values, and consecutive frames change by about 0.6. In the
cropped rerenders, the corresponding horizontal difference is 2.7–3.5 and
frame-to-frame change is about 1.5. These are diagnostics of the visible
border behavior, not a motion-quality score for the whole image.

The standalone renderer now defaults to `--guide-fit center_crop`: it scales
the real image to fill 512×832, then crops centrally. The resulting guide has
no repeated border columns. `--guide-fit edge_pad` is retained solely for
replaying old renders; it reproduced the old guide byte for byte in a dry run.
The cropped idle, talking, and smiling contact sheets show clean shoulder
edges. The crop changes face scale slightly, so old and new clips should be
compared as a motion test rather than mixed in one pose set.

## Head-motion experiments

The [measurement script](scripts/measure_pose_motion.py) checks every decoded
frame with FaceMesh. Eye-line displacement is a useful screening measure,
not a substitute for normal-speed visual review. The same script measures
the liked Indian native talking clip at 12.2 px of maximum vertical eye
travel, 8.8% of its initial eye distance. A small head movement is expected;
the problem is the conspicuous upward lift in the Japanese talking and smile.
The accepted Indian close-up smile measures 5.7 px or 4.1% across its shorter
145-frame, six-second clip, with a 0.6 px peak lip gap. Its length differs
from the requested ten-second Japanese smile, so this is a visual reference,
not a strict numeric gate.

| Version | Change | Talking vertical eye travel | Smile vertical eye travel | Other observation |
| --- | --- | ---: | ---: | --- |
| Prior selected | Padded guide, prior prompts | 32.7 px | 19.1 px | Stretch at side borders. |
| [v2](config/prompt_packs/japanese_native_three_pose_crop_level_head_v2.json) | Center crop; connected shoulder breathing; head level | 25.7 px | 23.2 px | Clean borders, but talking glances away and smile raises head. Idle measured 8.0 px and stayed calm. |
| [v3](config/prompt_packs/japanese_native_three_pose_crop_head_anchored_v3.json) | Same guide and seeds; stronger eye-line wording | 25.1 px | 22.6 px | Nearly the same head paths as v2. |
| [v4](config/prompt_packs/japanese_native_three_pose_crop_seed197_v4.json) | Same v3 text, seed 197 for talk and smile | 18.1 px | 13.8 px | Talking gaze is steadier; smile briefly parts lips (8.9 px peak gap). |
| [v5](config/prompt_packs/japanese_native_three_pose_crop_expression_anchored_v5.json) | Seed 197 and stronger positive instructions for level chin, relaxed neck, connected shoulders, and lips touching | 14.5 px | 13.8 px | Talking is close to the Indian native reference, with 30.2 px peak lip gap and steady gaze. Smile still shows teeth, with 11.5 px peak lip gap. |
| [v6](config/prompt_packs/japanese_native_three_pose_crop_smile_seed195_v6.json) | Same v5 smile text and guide, smile seed 197→195 | Not rerendered | 13.8 px | Closed lips (0.6 px peak gap), but a conspicuous sideways head tilt (9.9°) and 17.9 px lateral eye travel. Rejected. |
| [v7](config/prompt_packs/japanese_native_three_pose_crop_smile_seed193_v7.json) | Same v5 smile text and guide, return to seed 193 | Not rerendered | 23.2 px | Closed lips (0.4 px peak gap), but the middle of the smile still lifts the head. Rejected. |
| [v8](config/prompt_packs/japanese_native_three_pose_crop_smile_seed191_v8.json) | Same v5 smile text and guide, smile seed 193→191 | Not rerendered | 18.5 px | Closed lips (0.5 px peak gap), level shoulders, minimal sideways tilt (0.8°), but a brief upward head lift remains. |
| [v9](config/prompt_packs/japanese_native_three_pose_crop_smile_seed189_v9.json) | Same v5 smile text and guide, smile seed 191→189 | Not rerendered | 72.2 px | Deep downward nod and visible mouth opening (8.3 px peak gap). Rejected. |
| [v10](config/prompt_packs/japanese_native_three_pose_crop_smile_stationary_v10.json) | Seed 191 and same cropped guide; stronger positive prompt tying head to background | Not rerendered | 20.5 px | Closed lips (0.4 px gap), but head lift persists and is slightly greater than v8. Rejected. |

These measurements use complete 241-frame files. The first and last decoded
frames match exactly **within each** completed clip. The clips do not have a
shared endpoint across poses. The local rerun workspaces hold each iteration's
audit; the [selected review set](generated/portrait_pose_set_20260923/ltx/japanese_shoulders_head_review_v2/README.md)
includes the three accepted candidates, their exact graphs, and the final
validation. The Indian reference audit is
`generated/portrait_pose_set_20260923/ltx/indian_talking_motion_reference.json`.

The v2→v3 comparison shows that stronger wording alone did little under the
same seed. The v4 seed change reduced the lift but compromised the closed-lip
smile. The v5 talking wording reduced vertical eye travel to 9.5% of eye
distance, close to the Indian native talking clip's 8.8%, while preserving a
similar mouth opening (30.2 vs 30.3 px). The v5 smile still fails the lip
criterion. Seed 191 in v8 produced the best overall closed-lip smile of this
crop series, though its brief 18.5 px lift remains visible. The v10 prompt-only
attempt did not improve that lift. This review selects v8 as the most usable
candidate, not as proof that prompting can guarantee an Indian-V8-level stable
head. Human normal-speed acceptance remains necessary.
