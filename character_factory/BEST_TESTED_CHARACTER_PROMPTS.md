# Best-tested prompts for the current three-pose character pilot

The current Japanese best-tested **review candidates** use the exact text,
seeds, and idle repeat count in
[`japanese_selected_native_three_pose_v3.json`](config/prompt_packs/japanese_selected_native_three_pose_v3.json).
The [v3 review set](generated/portrait_pose_set_20260923/ltx/japanese_breath_speech_review_v3/README.md)
contains the videos and measurements. The selected idle is a 3.375-second
native LTX cycle repeated three times to make 10.04 seconds; talking is a
single 10.04-second native LTX render; smile is the unchanged v2 clip. These
are observations for this portrait, not guaranteed settings for other faces.
Human normal-speed approval and shared-anchor certification remain pending.

The previous
[`japanese_selected_native_three_pose_v2.json`](config/prompt_packs/japanese_selected_native_three_pose_v2.json)
pack and [v2 review videos](generated/portrait_pose_set_20260923/ltx/japanese_shoulders_head_review_v2/README.md)
remain frozen below. Their talk stopped at 3.92 seconds and idle had one
conspicuous deep breath. The [controlled reroll notes](JAPANESE_SHALLOW_BREATH_CONTINUOUS_TALK_20260924.md)
explain why v3, v4, and the long v5 idle were rejected and how the short-cycle
idle and v5 talking were selected.

## Portrait and LTX setup

Use the [gender-neutral FaceTime pose skeleton](generated/portrait_pose_set_20260923/pose_skeleton.png)
and [portrait template](config/prompt_packs/portrait_prompt_template.md) for
composition, then supply the character's identity, clothing, and room. The
[Japanese source portrait](generated/portrait_pose_set_20260923/japanese_woman.png)
was the input in this test. Preserve a centered, upright face, direct gaze,
closed relaxed mouth, and visible shoulders.

All selected clips use native LTX 2.3 22B distilled Q4_K_M, the accepted
low-VRAM eight-step Euler graph at CFG 1.0, 512×832 and 24 fps, with no
audio. Talking and smile are 241-frame native renders. Idle is an 81-frame
native render assembled into a 241-frame delivery video. All delivered clips
last 10.04 seconds. The same portrait guides native generation at frames `0`
and `-1` with strength 1. The runner verifies literal within-clip endpoint
equality and adds no fade or crossfade. There is no SoulX, Segmind, NAG,
Prompt Relay, or MuseTalk in these renders.

**Use `--guide-fit center_crop`.** The earlier `edge_pad` preprocessing
repeated roughly 22 columns on each side of this portrait, creating the
shoulder-edge smear. The corrected guide contains only real image pixels;
see the [before/after image](generated/portrait_pose_set_20260923/ltx/japanese_guide_fit_before_after.jpg).

## Current best-tested idle — seed 197, 81 frames repeated three times

Positive prompt:

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, with the same room, light, clothing, and framing. She quietly looks into the lens and keeps her head upright and level. Her expression and shoulder outline stay almost still. One small, natural breath gently moves only her upper chest and settles back into the starting pose. She gives one quick natural blink while her lips stay closed. Continuous shot, stable background.
```

Negative prompt:

```text
speaking, lip articulation, open mouth, visible teeth, distorted face, identity change, duplicate facial features, off-camera gaze, camera movement, scene cut, text, subtitles, watermark
```

Generate an 81-frame native LTX clip and repeat the endpoint-matched cycle
three times without duplicate join frames. The renderer honors
`"repeat_cycles": 3` in the v3 pack, or the already-rendered short clip can
be assembled with [the repeat script](scripts/repeat_short_idle_cycle.py).
The selected ten-second clip has three blinks, exact decoded loop joins, 5.2 px
maximum shoulder travel, 5.3 px eye-line travel, and closed lips. The v2 idle
had 14.1 px shoulder travel concentrated in one long rise.

## Current best-tested talking — seed 191, 241 frames

Positive prompt:

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, with the same room, light, clothing, and framing. She looks straight into the lens and speaks one lively, uninterrupted ten-second sentence. Her mouth and jaw keep forming new conversational syllables throughout the shot, including the final seconds; brief lip closures are part of the words, never a long silent pause. She stays engaged in the sentence until the last moment. Her head remains upright and level, her neck relaxed. Small, regular breaths happen naturally between words, with barely visible motion in her upper chest and shoulders. She gives two quick natural blinks. Continuous shot, stable background.
```

Negative prompt:

```text
silent person, frozen mouth, upward chin lift, neck extension, shoulder distortion, stretched frame borders, off-camera gaze, distorted face, identity change, duplicate facial features, camera movement, scene cut, text, subtitles, watermark
```

This clip has 6.3 px maximum shoulder travel and varied lip activity through
second 8. The last measured lip gap above 3 px occurs at 8.71 s; it then
settles for 1.25 s before the identical final frame. It fixes the v2
middle-of-shot silence, but it does **not** literally articulate through the
last frame.

## Current best-tested smiling — seed 191, unchanged from v2

The selected smile prompt and its exact text are in the frozen v2 section
below. It was not regenerated in the breathing and talking test.

## Frozen v2 baseline prompts

### Idle — seed 197

Positive prompt:

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, in the same room, lighting, clothing, and framing. She looks calmly into the lens with her head upright and level at the reference height. Gentle breathing creates small connected movement through her upper chest and shoulders, and she gives two quick natural blinks. Her lips stay comfortably closed and her jaw relaxed. She remains near her resting posture and ends in the exact starting pose. Continuous shot, stable background.
```

Negative prompt:

```text
speaking, lip articulation, open mouth, visible teeth, distorted face, identity change, duplicate facial features, off-camera gaze, camera movement, scene cut, text, subtitles, watermark
```

This prompt asks for connected breathing through upper chest and shoulders,
blinks, and an upright resting head. Its selected render has 8.0 px maximum
vertical eye movement and 0.5 px maximum central lip gap.

### Talking — seed 197

Positive prompt:

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, with the same room, light, clothing, and framing. She speaks calmly to the lens throughout the shot with clear varied conversational lip and jaw shapes and a couple of quick natural blinks. Her eyes stay on the lens, her eye line and chin remain level at the reference height, and her neck keeps its relaxed starting length. Gentle breathing moves her upper chest and both shoulders together while their natural shape stays intact. Her speech ends and she returns to the exact starting expression and pose. Continuous shot, stable background.
```

Negative prompt:

```text
silent person, frozen mouth, upward chin lift, neck extension, shoulder distortion, stretched frame borders, off-camera gaze, distorted face, identity change, duplicate facial features, camera movement, scene cut, text, subtitles, watermark
```

The accepted Indian native talking prompt motivated this behavior, but this
Japanese wording was revised after a direct gender-word substitution lifted
her head too much. The selected clip has 14.5 px maximum vertical eye
movement, 9.5% of its initial eye distance. The liked Indian native talking
clip measures 8.8% by the same method; peak central lip gap is 30.2 versus
30.3 px. The comparison is a motion diagnostic, not a guarantee of the same
perceived quality.

### Smiling — seed 191

Positive prompt:

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, with the same room, light, clothing, and framing. A restrained closed-lip smile slowly appears in the corners of her mouth and softly in her cheeks, holds, then fades back to neutral. Her upper and lower lips stay gently in contact through the entire smile. She looks into the lens and gives one quick natural blink. Her eye line and chin remain level at the reference height, her neck keeps its relaxed starting length, and gentle breathing moves her upper chest and both shoulders together. She ends in the exact starting expression and pose. Continuous shot, stable background.
```

Negative prompt:

```text
speaking, lip articulation, parted lips, visible teeth, upward chin lift, neck extension, shoulder distortion, stretched frame borders, distorted face, identity change, duplicate facial features, off-camera gaze, camera movement, scene cut, text, subtitles, watermark
```

Seed 191 was the strongest closed-lip option tested with this cropped guide:
0.5 px maximum central lip gap, 0.8° maximum change in eye-line tilt, and no
pronounced sideways sway. It still briefly lifts the head: maximum vertical
eye movement is 18.5 px, or 12.1% of eye distance. The accepted Indian V8
close-up smile measures 4.1% over a shorter six-second clip. More emphatic
prompt wording with the same seed did not remove the lift, so this smile
remains a review candidate rather than a finished motion match.

## Evidence and limits

The [idle](generated/portrait_pose_set_20260923/ltx/japanese_shoulders_head_review_v2/idle.mp4),
[talking](generated/portrait_pose_set_20260923/ltx/japanese_shoulders_head_review_v2/talking.mp4),
and [smiling](generated/portrait_pose_set_20260923/ltx/japanese_shoulders_head_review_v2/smiling.mp4)
clips each have 241 silent frames, an identical decoded first and last frame
within that clip, and clean cropped-guide shoulder edges. Their endpoint
hashes differ **across** clips, so they are not yet a switch-safe MuseTalk
pose set. The [validation](generated/portrait_pose_set_20260923/ltx/japanese_shoulders_head_review_v2/validation.json)
and [full rerun record](JAPANESE_SHOULDER_HEAD_RERUN_20260924.md) document
the checks and rejected attempts. The earlier prompt lineage remains in
[JAPANESE_NATIVE_THREE_POSE_PROMPTS_V1.md](JAPANESE_NATIVE_THREE_POSE_PROMPTS_V1.md).
