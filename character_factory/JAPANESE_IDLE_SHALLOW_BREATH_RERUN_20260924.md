# Japanese idle: visible shallow breathing without head wobble

The user accepted the talking and smiling clips in the
[v3 review set](generated/portrait_pose_set_20260923/ltx/japanese_breath_speech_review_v3/README.md)
but rejected its idle: the repeated short clip looks like rapid, millimeter-
scale head wobble and has no perceptible shallow breathing. **That idle is
rejected**, regardless of its earlier shoulder-landmark score. The existing
talking and smiling files remain untouched during this idle-only rerun.

The earlier audit measured shoulder landmarks but did not measure the shirt.
The new [neckline tracker](scripts/measure_idle_neckline.py) follows the first
shirt-colored pixel across the center neckline in every decoded frame. It is
calibrated for this high-contrast portrait, so its numbers are a diagnostic,
not a substitute for normal-speed viewing. Its
[baseline report](generated/portrait_pose_set_20260923/ltx/japanese_idle_neckline_baseline_audit.json)
shows the actual failure:

| Clip | Center neckline motion | What that means visually |
| --- | ---: | --- |
| Rejected v7 repeated idle | **0 px** | Shirt neckline is stationary across all 241 frames; small measured shoulder motion was not breathing. |
| Earlier v2 10-second idle | 28 px | One pronounced lift, the deep breath the user disliked. |
| Liked Indian V6 certified idle | About 2 px | Restrained cloth movement; useful scale reference, though it has different body proportions. |

Acceptance for the next Japanese idle requires a 10.04-second silent delivery
clip with exact decoded first/last equality, a clearly present but restrained
change in the shirt/upper chest over time, eyes open except quick blinks,
closed lips, and no repetitive head bob. A target neckline range of roughly
2–8 px is a diagnostic guide, not an automatic pass. Full-speed visual review
decides whether the breathing actually looks natural.

| Attempt | Change | Measured/observed result | Decision |
| --- | --- | --- | --- |
| [v10 prompt](config/prompt_packs/japanese_idle_chest_breath_seed195_v10.json) | Native 241 frames; make chest breathing the main action, seed 195. | One 24 px neckline rise; eyes near-closed for 6.46 s. | Reject. |
| [v11 prompt](config/prompt_packs/japanese_idle_neckline_breath_seed197_v11.json) | Native 121-frame breath cycle repeated twice, seed 197. | Each breath lifts neckline 26 px; lips part up to 11.5 px; eyes near-closed for 3.25 s per cycle. | Reject. |
| [v12 prompt](config/prompt_packs/japanese_idle_attentive_listening_seed193_v12.json) | Native 241 frames; attentive listening is the main behavior, breathing stays in background. | Neckline range 4 px, but eyes near-closed for 1.54 s, head shifts sideways and eye line tilts 5°. | Reject. |
| [v13 prompt](config/prompt_packs/japanese_idle_attentive_listening_seed199_v13.json) | Same v12 wording, seed 199. | Neckline range 20 px, mouth gap 14.2 px, eye-line tilt 12.5°. | Reject. |
| [v14 prompt](config/prompt_packs/japanese_idle_alert_centered_seed193_v14.json) | Return to seed 193 with alert eyes, centered level face, and quiet background breathing. | Neckline range 3 px and lips closed, but eyes near-closed for 1.71 s and head leans sideways. | Reject. |
| [v15 prompt](config/prompt_packs/japanese_idle_alert_centered_seed201_v15.json) | Exact v14 text with seed 201. | One 23 px neckline rise, 16.9 px shoulder rise, and mouth gap reaches 10 px. | Reject. |
| [v16 prompt](config/prompt_packs/japanese_idle_eyes_open_quiet_breath_seed193_v16.json) | Return to v12 seed 193, remove the requested blink, and emphasize alert open eyes. | Eyes close for at most 0.17 s; lips remain closed; shoulder travel is 5 px. Neckline range is just 3 px and mostly stationary by second. Head drifts sideways 15.8 px and tilts 5.8°. | Better face, but breathing is still likely too subtle; hold for comparison, not selected. |
| [v17 prompt](config/prompt_packs/japanese_idle_five_second_breath_seed193_v17.json) | Native 121-frame cycle repeated twice; one modest breath per five seconds, seed 193, no requested blink. | Two 12 px neckline rises, 2.6 px sideways head travel, 0.8° eye-line tilt, lips closed, longest eye closure 0.17 s. Six short blinks repeat with the cycle. | Strongest candidate; needs normal-speed human review. |

The [motion auditor](scripts/measure_pose_motion.py) now reports the longest
run of near-closed eyes relative to the open first frame. Its earlier blink
count alone mislabeled the v10/v11 prolonged eye closure as multiple blinks.
The [v17 review clip and audits](generated/portrait_pose_set_20260923/ltx/japanese_idle_breath_review_v4/README.md)
are preserved for human playback. It has an observable pair of smaller breaths
and a steadier head than v7. The repeated blink pattern may still look too
regular; **do not mark it accepted before normal-speed review**. The full
three-pose set still needs shared-anchor certification after idle approval.
Raw prompt-iteration workspaces remain local and ignored by Git.
