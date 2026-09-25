# Character creation workflow for the Lingua pilot

This is the current **one-avatar, three-pose** workflow. Start with one language
and one character. The 104-language roster and six-pose production pipeline in
[README.md](README.md) and [CODEX_PLAYBOOK.md](CODEX_PLAYBOOK.md) remain available,
but they are a different workflow. Do not start a bulk roster run to test one
new character.

The deliverables for a pilot character are one reviewed portrait and three
silent LTX 2.3 clips: idle, talking, and smiling. The selected prompt text,
provenance, settings, and Japanese test evidence live in the user-approved
[PERFECT_THREE_POSE_PROMPTS.md](PERFECT_THREE_POSE_PROMPTS.md). Historical
iterations remain in
[BEST_TESTED_CHARACTER_PROMPTS.md](BEST_TESTED_CHARACTER_PROMPTS.md). This workflow
creates review assets; it does not by itself create a switch-safe MuseTalk pose
bank or a prepared WebRTC avatar.

## 1. Create and approve one source portrait

Use the interactive Codex/ChatGPT image-generation tool, not an image API. For
the established FaceTime composition, use the gender-neutral
[pose skeleton](generated/portrait_pose_set_20260923/pose_skeleton.png) only as a
layout reference: upright centered head, eye-level camera, relaxed level
shoulders, upper-chest crop, direct gaze, and a closed mouth. The skeleton
must not impose gender or ethnicity on the new character. The reusable
[portrait template](config/prompt_packs/portrait_prompt_template.md) explains
the photographic and face-detection constraints. The Japanese portrait used
in this test is
[japanese_woman.png](generated/portrait_pose_set_20260923/japanese_woman.png).

Reject an image with an open mouth or visible teeth, an obscured or strongly
turned face, inconsistent framing, or a beauty-filter look. Review the image
before spending GPU time. Save the approved original PNG and record its path
and hash. For the standalone runner, the original does not need to be in the
roster or canonicalized by `ingest_portraits.py`; the runner prepares its own
512×832 guide without stretching the portrait.

## 2. Select the tested prompt pack
## 2. Select the approved prompt pack

The canonical Japanese pack is
[japanese_fixed_distance_shared_anchor_v1.json](config/prompt_packs/japanese_fixed_distance_shared_anchor_v1.json).
The user approved its idle, talking, and smiling trio as perfect after
normal-speed review on 2026-09-25. Its exact wording, seeds, graph settings,
hashes, and evidence are frozen in
[PERFECT_THREE_POSE_PROMPTS.md](PERFECT_THREE_POSE_PROMPTS.md). Use the pack
unchanged when reproducing that portrait.

For a different character, copy the approved pack to a new versioned JSON
file. Change only identity words and pronouns required by the new portrait;
keep the camera, distance, motion, breathing, expression, and negative-prompt
clauses unchanged for the first run. Keep the approved seeds as starting
seeds, then version any reroll instead of overwriting the approved pack.
Record the prompt, seed, portrait hash, settings, and visual reason for each
change. LTX is stochastic, so every new identity still needs normal-speed
review before its output is labeled approved.

## 3. Dry-run, then render all three poses

From /workspace/MuseTalk on the current machine:

~~~bash
LTX_PY=/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/.venv/bin/python
IMAGE=/workspace/MuseTalk/character_factory/generated/portrait_pose_set_20260923/japanese_woman.png
PACK=/workspace/MuseTalk/character_factory/config/prompt_packs/japanese_fixed_distance_shared_anchor_v1.json
OUT=/workspace/MuseTalk/character_factory/generated/<character_id>/ltx/approved_recipe_review

"$LTX_PY" character_factory/scripts/generate_three_pose_videos.py \
  --image "$IMAGE" --output-dir "$OUT" --prompt-pack "$PACK" \
  --guide-fit center_crop --shared-anchor --dry-run

"$LTX_PY" character_factory/scripts/generate_three_pose_videos.py \
  --image "$IMAGE" --output-dir "$OUT" --prompt-pack "$PACK" \
  --guide-fit center_crop --shared-anchor
~~~

Replace <character_id> and the image and copied-pack paths for a new identity.
The runner uses its configured low-VRAM ComfyUI environment and GPU lock. On a
different machine, adapt those paths in generate_three_pose_videos.py before
running. The dry run writes the prepared guide and three generation graphs.
Inspect graphs/*-generation.json to confirm the exact portrait, prompts,
seeds, 241-frame length, and guide indices 0 and -1.

The real run emits idle.mp4, talking.mp4, smiling.mp4, native source videos, a
manifest, contact sheets under review/, and generation/decode graphs. Each
delivered clip is 512 x 832, 24 fps, 241 frames (10.04 seconds), and silent.
All three are native 241-frame generations. The graph uses the same portrait
as both endpoint guides at strength 1, an eight-step Euler schedule, CFG 1,
and 64/16 temporal plus 256/64 spatial tiled VAE decode.

The --shared-anchor packaging step installs the same prepared portrait at the
first and last delivery frame of all three clips. It then decodes the encoded
videos and requires the boundary pixels to match both within and across clips.
It adds no fade or crossfade. SoulX, Segmind, Prompt Relay, NAG, MuseTalk, and
WebRTC rendering are not used in these source videos.

The `center_crop` guide fit is the current default. It fills the 512×832
canvas with real portrait pixels, cropping a small amount from the top and
bottom when the source is narrower than that ratio. Earlier Japanese tests
used `edge_pad`, which repeated the source's outermost pixel columns and
produced visibly stretched shoulder edges. Use `--guide-fit edge_pad` only to
replay one of those historical renders; its output guide has a different
hash and must not be mixed with the new crop in one pose set.

If a separate experimental workflow shows face ghosting or flicker, check its
decode graph before changing the prompt. The [September 25 controlled test](LTX23_JAPANESE_FLICKER_VALIDATION_20260925.md)
reproduced the problem with temporal `16/8` and improved the same saved latent
at `64/16`. This runner already uses the corrected profile; preserve it.

## 4. Review motion and mouth behavior before accepting a clip

Play each MP4 at normal speed, then inspect any suspect moment frame by
frame. Contact sheets and landmarks help locate defects but cannot replace
playback. Check the same identity, clothing, framing, and stable background
throughout all three clips. In particular:

For this Japanese portrait, run `scripts/measure_idle_neckline.py` on each
idle candidate as a shirt-motion diagnostic. The rejected v3 loop measures
0 px there despite small shoulder-landmark movement; the earlier deep-breath
clip measures 28 px. The tracker uses this portrait's dark-shirt/skin boundary
and must be recalibrated before use on a different portrait.

| Pose | Required observation | Failure examples from this test |
|---|---|---|
| Idle | Calm direct gaze, natural blinks, several shallow breaths, closed mouth, head near its starting height | The v3 short cycle visibly wobbles the head with no shirt-level breathing; several longer rerolls make one deep inhale or prolonged eye closure. No Japanese idle has passed review. |
| Talking | Natural conversational lip shapes throughout the clip without a camera move or repeated mechanical sway | V2 went quiet at 3.92 s; the current candidate speaks into second 8 but still has a 1.25 s quiet tail. |
| Smiling | Modest smile, lips together, return to neutral, no large chin lift | Seed 197 opened the mouth; seed 195 tilted the head sideways; the selected seed 191 still has a brief upward lift. |

Record the accepted/rejected decision for each clip with its prompt-pack ID,
seed, file hash, and visual reason. For a reroll, change one pose at a time
with `--poses idle`, `--poses talking`, or `--poses smiling`, and use a new
output directory and versioned prompt pack. Keep rejected clips and their
manifests long enough to explain why the selected version won. Do not use
`--force` to overwrite a reviewed render unless replacement is intentional.

The Japanese [historical review set](generated/portrait_pose_set_20260923/ltx/japanese_breath_speech_review_v3/README.md)
shows the accepted talking/smiling clips and rejected idle. Its
[`validation.json`](generated/portrait_pose_set_20260923/ltx/japanese_breath_speech_review_v3/validation.json)
checks every decoded frame and independently verifies the end frames.

## 5. Keep the endpoint and integration gates separate

An identical first and last frame **inside one video** only makes that clip
loop to itself. It does not make the three clips interchangeable. The Japanese
review clips have three different endpoint RGB hashes, so a direct pose
switch could visibly jump. Before using them as a MuseTalk pose set, run a
shared-anchor certification step across all selected clips and re-decode to
verify the common boundary. The existing six-pose roster certifier is not a
drop-in three-pose packager; adapt and test that path rather than treating the
standalone renderer's manifest as a runtime pose manifest.

Only after visual acceptance and common-anchor certification should a
three-pose runtime manifest, MuseTalk caches, and WebRTC session be prepared
and tested. The September Japanese prompt test stopped before those steps.
Keep the roster's `config/languages.json` seed separate until Lingua's pilot
language IDs are settled; it is not required to render one portrait here.
