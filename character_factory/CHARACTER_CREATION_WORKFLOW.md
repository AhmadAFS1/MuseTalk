# Character creation workflow for the Lingua pilot

This is the current **one-avatar, three-pose** workflow. Start with one language
and one character. The 104-language roster and six-pose production pipeline in
[README.md](README.md) and [CODEX_PLAYBOOK.md](CODEX_PLAYBOOK.md) remain available,
but they are a different workflow. Do not start a bulk roster run to test one
new character.

The deliverables for a pilot character are one reviewed portrait and three
silent LTX 2.3 clips: idle, talking, and smiling. The selected prompt text,
provenance, and Japanese test evidence live in
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

For the tested Japanese woman, use
[`japanese_selected_native_three_pose_v2.json`](config/prompt_packs/japanese_selected_native_three_pose_v2.json).
It selects idle seed 197, talking seed 197, and smile seed 191. The talking
prompt was revised from the liked Indian native-LTX wording because direct
gender-word substitution made this Japanese portrait lift her head. The new
smile is closed-lip and avoids sideways sway, but it still has a brief upward
head lift. These are the best **observed review candidates** for this portrait,
not universal or visually approved prompts for every face.

For a different character, copy the selected pack to a new, versioned JSON
file. Change identity words and pronouns deliberately while keeping one
behavioral idea per pose. Keep the original selected pack unchanged so the
Japanese render manifest remains reproducible. Record every changed prompt,
seed, and visual reason in the new character's review notes. Do not silently
substitute the script's historical default pack for the selected one.

## 3. Dry-run, then render all three poses

From `/workspace/MuseTalk` on the current machine:

```bash
LTX_PY=/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/.venv/bin/python
IMAGE=/workspace/MuseTalk/character_factory/generated/portrait_pose_set_20260923/japanese_woman.png
PACK=/workspace/MuseTalk/character_factory/config/prompt_packs/japanese_selected_native_three_pose_v2.json
OUT=/workspace/MuseTalk/character_factory/generated/<character_id>/ltx/native_three_pose_v2

"$LTX_PY" character_factory/scripts/generate_three_pose_videos.py \
  --image "$IMAGE" --output-dir "$OUT" --prompt-pack "$PACK" \
  --guide-fit center_crop --dry-run

"$LTX_PY" character_factory/scripts/generate_three_pose_videos.py \
  --image "$IMAGE" --output-dir "$OUT" --prompt-pack "$PACK" \
  --guide-fit center_crop
```

Replace `<character_id>` and the image and pack paths for a new character.
The runner uses its configured low-VRAM ComfyUI environment and GPU lock. On a
different machine, those paths in `generate_three_pose_videos.py` must be
adapted before running. The dry run writes the prepared guide and three
generation graphs. Inspect `graphs/*-generation.json` to confirm the exact
portrait, prompts, seeds, 241-frame length, and guide indices `0` and `-1`.

The real run emits `idle.mp4`, `talking.mp4`, `smiling.mp4`, a `manifest.json`,
contact sheets under `review/`, and generation/decode graphs. Each clip is
512×832, 24 fps, 241 frames (10.04 seconds), and silent. The native graph uses
the same portrait as both endpoint guides at strength 1, an eight-step Euler
schedule, and 64/16 tiled VAE decode. The delivery encode replaces the final
decoded frame with the first and verifies exact decoded first/last equality
**within each clip**. It does not add a fade or crossfade. SoulX, Segmind,
Prompt Relay, and NAG are not used.

The `center_crop` guide fit is the current default. It fills the 512×832
canvas with real portrait pixels, cropping a small amount from the top and
bottom when the source is narrower than that ratio. Earlier Japanese tests
used `edge_pad`, which repeated the source's outermost pixel columns and
produced visibly stretched shoulder edges. Use `--guide-fit edge_pad` only to
replay one of those historical renders; its output guide has a different
hash and must not be mixed with the new crop in one pose set.

## 4. Review motion and mouth behavior before accepting a clip

Play each MP4 at normal speed, then inspect any suspect moment frame by
frame. Contact sheets and landmarks help locate defects but cannot replace
playback. Check the same identity, clothing, framing, and stable background
throughout all three clips. In particular:

| Pose | Required observation | Failure examples from this test |
|---|---|---|
| Idle | Calm direct gaze, natural blinks, subtle breathing, closed mouth, head near its starting height | The first Japanese idle dropped its head deeply despite asking for a small posture adjustment. |
| Talking | Natural conversational lip shapes without a camera move or repeated mechanical sway | The first Japanese talking clip articulated clearly but lifted her head far more than the liked Indian native talking clip. |
| Smiling | Modest smile, lips together, return to neutral, no large chin lift | Seed 197 opened the mouth; seed 195 tilted the head sideways; the selected seed 191 still has a brief upward lift. |

Record the accepted/rejected decision for each clip with its prompt-pack ID,
seed, file hash, and visual reason. For a reroll, change one pose at a time
with `--poses idle`, `--poses talking`, or `--poses smiling`, and use a new
output directory and versioned prompt pack. Keep rejected clips and their
manifests long enough to explain why the selected version won. Do not use
`--force` to overwrite a reviewed render unless replacement is intentional.

The Japanese [review set](generated/portrait_pose_set_20260923/ltx/japanese_shoulders_head_review_v2/README.md)
shows the selected videos and contact sheets. Its
[`validation.json`](generated/portrait_pose_set_20260923/ltx/japanese_shoulders_head_review_v2/validation.json)
checks every decoded frame and independently verifies the end frames.

## 5. Keep the endpoint and integration gates separate

An identical first and last frame **inside one video** only makes that clip
loop to itself. It does not make the three clips interchangeable. The selected
Japanese clips have three different endpoint RGB hashes, so a direct pose
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
