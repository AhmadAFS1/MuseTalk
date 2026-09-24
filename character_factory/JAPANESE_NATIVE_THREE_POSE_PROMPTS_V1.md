# Japanese avatar: best-supported native LTX 2.3 prompts and tests

This sheet records the exact text used for the Japanese woman in
`generated/portrait_pose_set_20260923/japanese_woman.png`. It separates an
actually liked prompt from two new candidates. The approved Indian production
bank names V6 listening, V14/V15 speaking, and V8 smile as its winners, but the
original V6/V8/V14/V15 prompt-pack JSON files are absent from this checkout.
Their certified videos survive; their full prompt text does not. The exact
prompt for the later liked *native LTX talking* test does survive in
`/workspace/experiments/ltx23_native_flf_talking_20260922/graphs/generation.json`.
It is the strongest reproducible textual baseline here.

## Current best tested selection

The runnable selection is
`config/prompt_packs/japanese_selected_native_three_pose_v1.json`:

| Pose | Prompt in this sheet | Seed | Japanese clip | Result |
|---|---|---:|---|---|
| Idle | Idle candidate 2 | 197 | `ltx/japanese_idle_locked_candidate_v2/idle.mp4` | No deep nod; 6.3 px maximum vertical eye shift over all frames. |
| Talking | Liked Indian native talking prompt, gender words changed | 191 | `ltx/japanese_best_supported_native_v1/talking.mp4` | Clear articulation; 32.2 px maximum vertical eye shift. |
| Smiling | Smile candidate 3, V4 wording | 193 | `ltx/japanese_smile_v4_lineage_candidate_v3/smiling.mp4` | Modest closed-lip smile; 0.5 px maximum central lip gap, 18.6 px maximum vertical eye shift. |

The clip paths in this table are relative to
`generated/portrait_pose_set_20260923/`. Each one is a ten-second-class
silent MP4 whose decoded first and last frames match. The three endpoint
frames **do not match each other**. This selection is for visual review, not
for seamless MuseTalk pose switching. Normal-speed human acceptance is still
pending. A single review folder with links to the selected MP4s and contact
sheets is `generated/portrait_pose_set_20260923/ltx/japanese_selected_review_v1/`.
Independent full-frame checks are in
`generated/portrait_pose_set_20260923/ltx/japanese_selected_review_v1/validation.json`.

## Workflow held constant

- LTX 2.3 22B distilled Q4_K_M, native eight-step Euler graph, CFG 1.0.
- One Japanese portrait supplied at guide frames 0 and -1, both at strength 1.
- 512 × 832, 24 fps, 241 frames (10.04 seconds), no supplied audio.
- Decode with 64/16 tiled VAE. Replace the last decoded frame with the first,
  then encode all-intra H.264 and verify the decoded end frames match.
- No SoulX, Segmind, Prompt Relay, NAG, MuseTalk, or crossfade in this test.
- The negative prompts exclude technical defects and unwanted mouth behavior;
  they do not include broad motion-amplitude language.

The initial test used `config/prompt_packs/japanese_best_supported_native_v1.json`.
The current selection above combines later, tested rerolls without altering
their source prompt packs or videos.

## Idle — restrained candidate, seed 189

The approved Indian V6 clip shows gentle attention, blinks, and restrained
motion. Its exact prompt is unavailable. This new candidate borrows the
near-still breathing and blink behavior from surviving early Indian pose
manifests. It removes the earlier Japanese test's generic “conversational head
and shoulder movement” instruction, which produced a conspicuous downward nod.

**Positive prompt**

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, in the same room, lighting, clothing, and framing. She faces the camera in a relaxed upright resting pose, breathing gently with a subtle rise and fall in the upper chest. She gives one or two natural blinks at separated moments. Her head stays close to the starting position with only a small posture-settling adjustment. Her lips remain comfortably closed and her jaw relaxed. She settles naturally back into the exact starting pose at the end. Continuous shot, stable background.
```

**Negative prompt**

```text
speaking, lip articulation, open mouth, visible teeth, distorted face, identity change, duplicate facial features, off-camera gaze, camera movement, scene cut, text, subtitles, watermark
```

## Talking — liked Indian native prompt, identity words changed, seed 191

This is the September 22 Indian talking prompt with only `male/He/his`
changed to `female/She/her`. The user liked that Indian talking motion. Its
behavior on this Japanese portrait still needs visual review.

**Positive prompt**

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, in the same room, lighting, clothing, and framing. She speaks naturally and continuously to the camera with clear conversational lip articulation, natural blinking, and subtle head and shoulder movement. Preserve her facial proportions and identity. She settles naturally back into the exact starting pose at the end. Continuous shot, stable background.
```

**Negative prompt**

```text
silent person, frozen mouth, distorted face, identity change, duplicate facial features, off-camera gaze, camera movement, scene cut, text, subtitles, watermark
```

## Smiling — restrained candidate, seed 193

The approved Indian V8 clip has a moderate closed-lip smile. Its exact prompt
is unavailable. This new candidate follows that observed behavior and removes
the chin-lift instruction from the earlier Japanese test, whose smile raised
the head too much.

**Positive prompt**

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, in the same room, lighting, clothing, and framing. She faces the camera in a relaxed upright resting pose. A mild closed-lip smile gradually appears at the corners of her mouth and softly around her eyes, holds briefly, then fades back to a neutral expression. Her head and shoulders remain close to their starting position while she breathes gently and blinks naturally. Her lips stay together with no visible teeth. She settles naturally back into the exact starting pose at the end. Continuous shot, stable background.
```

**Negative prompt**

```text
speaking, lip articulation, open mouth, visible teeth, distorted face, identity change, duplicate facial features, off-camera gaze, camera movement, scene cut, text, subtitles, watermark
```

## Test and review gates

1. Render the three isolated clips with the prompt pack above.
2. Verify each delivery MP4 has 241 frames at 24 fps, contains no audio, and
   decodes to identical first and last RGB frames.
3. Review each clip at normal playback speed for face/identity continuity,
   camera stability, and excess nod, sway, or chin lift. Idle must stay gently
   engaged, talking must articulate naturally, and smile must be moderate and
   return to neutral.
4. Compare the new Japanese contact sheets against the previous Japanese
   candidate. Report defects honestly; endpoint equality alone is not a motion
   quality pass. These clips are a prompt test, not a production pose bank.

## First Japanese test: observed result

The first render is in
`generated/portrait_pose_set_20260923/ltx/japanese_best_supported_native_v1/`.
All three MP4s have 241 decoded frames at 24 fps, no audio, and an exact
decoded first/last match *within each clip*, as recorded in `manifest.json`.
The endpoint RGB hashes differ across the three clips, so this test is not a
certified switchable pose bank.

The 1 fps contact sheets and sampled face landmarks show:

| Pose | First Japanese test | Previous Japanese test | Approved Indian reference | Assessment |
|---|---:|---:|---:|---|
| Idle | 93.4 px eye-line vertical excursion | 101.2 px | 3.4 px | Failed: pronounced downward nod persists. |
| Talking | 30.7 px | 27.2 px | 10.6 px for V14 closed-mouth plate | Clear lip articulation, but head travel still exceeds the accepted Indian baseline; not approved. |
| Smile | 22.4 px | 42.0 px | 4.6 px | Improved, but still has vertical head travel and briefly reveals teeth despite closed-lip wording. |

Measurements use MediaPipe FaceMesh eye landmarks 33/263 on every twelfth
decoded frame. They are a diagnostic, not a substitute for normal-speed human
review. See `review/motion_evidence.json` in the test output. The Indian V14
speaking clip is a silent, closed-mouth MuseTalk body plate, whereas the new
Japanese talking clip intentionally articulates; their lip-gap values are not
comparable.

## Idle candidate 2, seed 197

Because candidate 1 still makes a deep nod, the second test removes its
“posture-settling adjustment” and keeps the head upright and centered. It is
stored separately in `config/prompt_packs/japanese_best_supported_native_idle_v2.json`.

**Positive prompt**

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, in the same room, lighting, clothing, and framing. She looks calmly into the lens with her head upright and centered for the entire shot. Gentle upper-chest breathing and two quick natural blinks occur while she holds her steady resting posture and keeps her chin level. Her lips stay gently closed and her jaw relaxed. She ends in the exact starting pose. Continuous shot, stable background.
```

**Negative prompt**: same as idle candidate 1 above.

This reroll is a test of a new prompt *and* a new seed; it cannot isolate which
change causes any improvement.

**Observed idle result:**
`generated/portrait_pose_set_20260923/ltx/japanese_idle_locked_candidate_v2/idle.mp4`
contains 241 frames at 24 fps with no audio and an exact decoded first/last
match. The sampled eye midpoint moves at most 5.0 px vertically and 11.8 px
horizontally; eye-line roll changes at most 1.1°. The contact sheet shows a
steady, blinking face without candidate 1's deep nod. This is the best tested
Japanese **idle** in this series, subject to normal-speed human review.

## Smile candidate 2, seed 199

Because candidate 1 briefly shows teeth and still raises the head, this
candidate makes lip contact and head position explicit. It remains inspired by
the preserved V4 small-smile description and approved V8 visual behavior; it
is not claimed to be the missing exact V8 text. It is stored in
`config/prompt_packs/japanese_best_supported_native_smile_v2.json`.

**Positive prompt**

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, in the same room, lighting, clothing, and framing. She looks into the lens with her head upright and centered. A small genuine smile gradually appears at the corners of her mouth and softly around her eyes, holds briefly, then fades back to the original neutral expression. Her upper and lower lips remain touching throughout, with no teeth visible. One quick natural blink and gentle upper-chest breathing continue while her head and shoulders stay in the starting position. Continuous shot, stable background.
```

**Negative prompt**: same as smile candidate 1 above. This is also a prompt and
seed change together, not a controlled single-variable comparison.

**Observed smile candidate 2 result:**
`generated/portrait_pose_set_20260923/ltx/japanese_smile_closed_candidate_v2/smiling.mp4`
contains the expected 241 silent frames and matching decoded endpoints, but
the mouth opens visibly around frame 70. Full-frame landmark review found a
21.4 px peak central lip opening and 25.7 px vertical eye excursion. It is
worse than candidate 1 and rejected.

## Smile candidate 3, seed 193

This final reroll holds the original smile seed at 193 and borrows the
surviving **V4** Indian small-smile action wording more directly. V4 was an
earlier candidate, not the winning V8. The prompt is stored in
`config/prompt_packs/japanese_best_supported_native_smile_v3.json`.

**Positive prompt**

```text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, in the same room, lighting, clothing, and framing. A small genuine closed-lip smile gradually appears at the mouth corners and softly around the eyes, as if warmly acknowledging the caller. Gentle breathing and one quick blink continue naturally. Her head stays comfortably stable while the smile holds. The smile remains modest, never showing teeth or becoming a broad camera pose, then fades smoothly to the reference expression. She settles back into the exact starting pose at the end. Continuous shot, stable background.
```

**Negative prompt**: same as smile candidate 1 above.

**Observed smile candidate 3 result:**
`generated/portrait_pose_set_20260923/ltx/japanese_smile_v4_lineage_candidate_v3/smiling.mp4`
has 241 silent frames and an exact decoded first/last match. Face landmarks
were detected in every decoded frame; the maximum central lip gap was 0.5 px,
while the smile widened the mouth by about 22 px from its resting width. The
maximum vertical eye-line excursion was 18.6 px. The contact sheet shows a
closed-lip smile that returns to neutral. This is the best tested Japanese
smile, but its vertical head motion remains greater than the approved Indian
V8 clip and needs normal-speed human review.

## Selected test evidence and limits

The selected clips are [idle](generated/portrait_pose_set_20260923/ltx/japanese_idle_locked_candidate_v2/idle.mp4),
[talking](generated/portrait_pose_set_20260923/ltx/japanese_best_supported_native_v1/talking.mp4),
and [smiling](generated/portrait_pose_set_20260923/ltx/japanese_smile_v4_lineage_candidate_v3/smiling.mp4).
The independent validation JSON records all 241 frame checks, video stream
properties, file hashes, head movement, mouth opening, and first/last RGB
equality. Contact sheets and detailed generation graphs remain next to each
MP4. No MuseTalk render, WebRTC session, or common-anchor certification was
performed in this prompt test. Sampled images and landmark data support the
assessment above; the clips should be judged at normal playback speed by a
human before any production use.
