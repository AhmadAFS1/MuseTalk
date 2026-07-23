# Segmind FaceTime Motion Reference Set

## Outcome

The three July 22 mannequin clips have good motion and exact shared neutral
anchors, but their original 1080x1920 framing is too wide for the existing AI
FaceTime avatars. In the raw clips, the head is smaller and sits higher than in
the recent avatar source images and Segmind outputs. Sending the raw framing
would make body-position transfer less predictable and could introduce a scale
or crop correction during generation.

All three clips therefore use one identical edit:

- 1.25x centered scale
- 60 px downward translation in the final 720x1280 frame
- top fill matched to the source background (`#EDF1F7`)
- 720x1280, 30 FPS, H.264/yuv420p, silent, 10 seconds

The normalized assets are in:

```text
assets/segmind_motion_references/facetime_v1/
  neutral_resting.mp4
  nod_agree.mp4
  look_away_reset.mp4
  manifest.json
  validation_report.json
```

The adjustment brings the crown, eye line, chin, shoulders, and upper-torso
coverage into the established close-up FaceTime composition. It does not alter
gesture timing. The full nod and left/right head turn remain inside frame.

## Validation Result

The source motions are appropriate for the first three-pose bank:

| Pose | Result | Notes |
| --- | --- | --- |
| `neutral_resting` | Pass | Stable centered shoulders and subtle breathing only. |
| `nod_agree` | Pass | Controlled vertical nod; no large torso displacement. |
| `look_away_reset` | Pass | Controlled head turn; shoulders remain near the canonical anchor. |

The raw clips use the exact same first and last neutral frame. Their normalized
H.264 encodes retain a near-identical loop anchor (SSIM 0.994), and every clip
has the exact same decoded first frame. The validator checks codec, resolution,
frame rate, duration, absence of audio, each clip's first/last-frame similarity,
and cross-clip first-frame similarity:

```bash
python3 -m scripts.validate_segmind_motion_references facetime_v1 \
  --report assets/segmind_motion_references/facetime_v1/validation_report.json
```

## Segmind Integration

`POST /avatars/generate` now accepts:

```json
{
  "motion_reference_preset": "facetime_v1"
}
```

The generation path uploads the checked-in references to the configured public
motion-reference bucket, invokes Kling 2.6 Pro Motion Control once per pose,
normalizes and optionally uploads each output, and returns all pose results in
`motion_videos`. `neutral_resting` is the default result and remains available
through the original `motion_video_path` and `motion_video_url` response fields,
so existing preparation logic continues to prepare the neutral pose.

The preset explicitly sends:

```text
character_orientation = video
keep_original_sound = false
```

This preserves the reference head-turn direction and avoids carrying unused
reference audio. The preset is opt-in because it makes three billable Segmind
requests rather than silently changing the cost of existing single-motion
avatar generation.

Generating the three non-neutral outputs does not automatically register them
as prepared MuseTalk speaking poses. If either gesture will be used during live
speech, prepare that output separately and register it through the existing
`prepared-poses` API; the prior multi-pose experiment showed that reusing
neutral face geometry across larger head changes creates visible drift. For
idle/listening/reaction playback, they can use the existing idle-pose upload
path without additional MuseTalk preparation.
