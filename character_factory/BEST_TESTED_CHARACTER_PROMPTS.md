# Best-tested prompts for the current three-pose character pilot

These are the exact prompts selected after six native LTX 2.3 renders of the
[Japanese portrait](generated/portrait_pose_set_20260923/japanese_woman.png).
“Best-tested” means strongest observed in this test, not a guarantee for every
avatar. Normal-speed human acceptance and shared-anchor certification are
still pending. Use the machine-readable
[selected prompt pack](config/prompt_packs/japanese_selected_native_three_pose_v1.json)
with the [workflow](CHARACTER_CREATION_WORKFLOW.md); keep this page as the
readable record of the text and why it was selected.

## Portrait before motion

Use the interactive image-generation workflow and the
[portrait template](config/prompt_packs/portrait_prompt_template.md). The
tested Japanese portrait followed the reusable
[gender-neutral skeleton](generated/portrait_pose_set_20260923/pose_skeleton.png):
vertical FaceTime crop, eye-level camera, head upright and centered, both
shoulders visible, direct gaze, closed relaxed lips, and hands outside the
frame. Its identity, wardrobe, and room details are recorded in
[PROMPTS.md](generated/portrait_pose_set_20260923/PROMPTS.md). For a new
character, change those identity details while preserving the camera and pose
constraints. Do not encode “Japanese woman” as a universal skeleton property.

## Native LTX settings common to all three

- Model: LTX 2.3 22B distilled Q4_K_M; native eight-step Euler graph, CFG 1.0.
- 512×832, 24 fps, 241 frames (10.04 seconds), no supplied audio.
- The same source portrait at first and last guide indices `0` and `-1`, both
  strength 1; 64/16 tiled VAE decode.
- The delivery encoder replaces the last decoded frame with the first and
  verifies exact decoded RGB equality within each clip. It does not certify
  that different clips share one endpoint.
- No SoulX, Segmind, Prompt Relay, NAG, or MuseTalk in these renders.

The common idle/smile **negative prompt** is:

```text
speaking, lip articulation, open mouth, visible teeth, distorted face, identity change, duplicate facial features, off-camera gaze, camera movement, scene cut, text, subtitles, watermark
```

The talking **negative prompt** is:

```text
silent person, frozen mouth, distorted face, identity change, duplicate facial features, off-camera gaze, camera movement, scene cut, text, subtitles, watermark
```

## Idle: selected Japanese candidate, seed 197

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, in the same room, lighting, clothing, and framing. She looks calmly into the lens with her head upright and centered for the entire shot. Gentle upper-chest breathing and two quick natural blinks occur while she holds her steady resting posture and keeps her chin level. Her lips stay gently closed and her jaw relaxed. She ends in the exact starting pose. Continuous shot, stable background.
```

This wording **was changed during the Japanese test**. The first candidate's
“small posture-settling adjustment” became a pronounced downward nod, with
93.4 px sampled vertical eye travel. The revised seed-197 candidate removed
that invitation and held the head upright; its full-frame maximum was 6.3 px.
It is informed by the approved Indian V6 listening behavior and surviving
older Indian neutral prompts, **not** a recovered verbatim V6 prompt.

## Talking: liked Indian native prompt, seed 191

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, in the same room, lighting, clothing, and framing. She speaks naturally and continuously to the camera with clear conversational lip articulation, natural blinking, and subtle head and shoulder movement. Preserve her facial proportions and identity. She settles naturally back into the exact starting pose at the end. Continuous shot, stable background.
```

The exact positive and negative prompts survive in the September 22 Indian
native-LTX graph at
`/workspace/experiments/ltx23_native_flf_talking_20260922/graphs/generation.json`.
The positive text above changes **only** `male/He/his` to `female/She/her`.
The user liked the Indian talking motion. On the Japanese portrait, the clip
shows clear articulation; full-frame vertical eye travel was 32.2 px. This
native talking prompt is separate from the older closed-mouth V14/V15 body
plates that MuseTalk was expected to animate.

## Smiling: V4-derived Japanese candidate, seed 193

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, in the same room, lighting, clothing, and framing. A small genuine closed-lip smile gradually appears at the mouth corners and softly around the eyes, as if warmly acknowledging the caller. Gentle breathing and one quick blink continue naturally. Her head stays comfortably stable while the smile holds. The smile remains modest, never showing teeth or becoming a broad camera pose, then fades smoothly to the reference expression. She settles back into the exact starting pose at the end. Continuous shot, stable background.
```

This wording **was changed during the Japanese test**. The first smile briefly
showed teeth (8.4 px peak measured lip gap); an attempt that added stricter
lip-contact language but changed the seed opened even wider (21.4 px). The
selected candidate returned to seed 193 and followed the surviving Indian V4
small-smile action wording more closely. Its maximum measured lip gap was
0.5 px across all 241 frames; vertical eye travel was 18.6 px. It is the best
tested Japanese smile, but its head movement exceeds the approved Indian V8
clip. The exact winning V8 prompt JSON is missing from this checkout; this
text is **not** claimed to be V8 verbatim.

## Test evidence and next use

The selected [idle](generated/portrait_pose_set_20260923/ltx/japanese_selected_review_v1/idle.mp4),
[talking](generated/portrait_pose_set_20260923/ltx/japanese_selected_review_v1/talking.mp4),
and [smiling](generated/portrait_pose_set_20260923/ltx/japanese_selected_review_v1/smiling.mp4)
videos, contact sheets, and [validation](generated/portrait_pose_set_20260923/ltx/japanese_selected_review_v1/validation.json)
are together in the Japanese review folder. Every selected clip has 241
silent frames and identical first/last decoded pixels within itself. Their
endpoint hashes differ **across** clips. Inspect them at normal playback speed
before treating any prompt as final, and certify a shared anchor before
switching poses in MuseTalk.

The full iteration record, including rejected prompt text, seeds, and visual
reasons, is in
[JAPANESE_NATIVE_THREE_POSE_PROMPTS_V1.md](JAPANESE_NATIVE_THREE_POSE_PROMPTS_V1.md).
