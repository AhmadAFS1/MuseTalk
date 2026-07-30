# Smooth speaking v3 — ARDY → Segmind → MuseTalk validation

Date: 2026-07-30  
Server: MuseTalk and ARDY co-located on `50.40.184.100:60523`

## Verdict

The replacement is a substantial improvement and the end-to-end WebRTC run passed, but the four Segmind-derived videos are **not mathematically identical at their switch boundaries**.

- ARDY source delivery: **PASS** — all 12 ordered transitions share the same decoded 15-frame handles.
- Segmind derivatives: **FAIL exact equality** — 0/6 first-frame pairs and 0/12 directed final-frame → first-frame transitions are pixel-identical.
- MuseTalk WebRTC runtime: **PASS** — 30 fps playback, 15 fps lip-sync, two-frame crossfade, all 12 ordered idle transitions and both TTS flows completed.
- Visual judgment: the earlier conspicuous jumps into `speaking_direct` are greatly reduced. Small residual changes remain in brow/eye state and vertical face placement.

Keep the two-frame crossfade enabled. Do not describe the current Segmind assets as mathematically seamless.

## ARDY remake

Only `speaking_direct_v2` was changed:

- Removed the early negative head anticipation.
- Scaled authored Head and Neck rotation to 20%.
- Preserved timing, shoulders, arms, spine, ambient motion, camera, framing, and the 15-frame canonical handles.
- Reduced maximum global head rotation from `1.917773°/frame` to `0.482092°/frame`.
- ARDY precision/video test result: 43 passed, 2 skipped.

Certified delivery:

`/workspace/ardy/outputs/pose_delivery_mvp_four_v3_smooth_speaking`

- Speaking input SHA-256: `f6ba6a95053cf9ff7746995bffcc81e5c614a2a62957a7349f57cbef79092b8b`
- Delivery manifest SHA-256: `61bb0066993808017a52c77946c6cd331461ade6dfb07d8a2c590fdd73361bf8`
- Delivery validation SHA-256: `de6b05901852077f2f121440a90584b5f4c2bbd160d34ee8e2a9e03ddf2d4cd6`
- Shared decoded 15-frame handle SHA-256: `0dc971c161da83d74dbf90bd0660626ec5d65a0c5396251e6e9bd646c29f652c`

## Segmind generation

Exactly one new paid request was submitted.

- Model: `kling-2.6-pro-motion-control`
- Request ID: `c42b69f1b13e8cca042ee9863104dd96`
- Cost: `$1.40`
- Character: unchanged Indian tutor image, SHA-256 `72b1ac0431025ceaf0620bf67ea9d448dba4376537c7965b134a89fa6f560d6b`
- Prompt: `Transfer the reference video speaking direct v2 motion to the character in the source image. Preserve the character identity and framing. Keep the natural blinking from the reference and maintain relaxed, direct eye contact with the camera whenever the eyes are open. Keep the camera locked and the mouth closed. No looking sideways.`

Installed MuseTalk asset:

`/workspace/MuseTalk/generated/downloads/indian_tutor_essential_six_v1/speaking_direct.mp4`

- Output SHA-256: `ee4f754078ae49c7adb77043605bd51c84f8fba8681e2e0149988478991b6b9e`
- Media: 720×1280, 30 fps, 300 frames, 10.0 seconds
- First/last SSIM: `0.820239`
- Prepared cache input hash matched the installed file.

The previous runtime asset is preserved at:

`/workspace/MuseTalk/generated/asset_analysis/2026-07-30_smooth_speaking_v3/speaking_direct_previous.mp4`

## Geometry results

First-frame DWPose landmark RMSE versus neutral after compensating for translation, rotation, and scale:

| Region | Previous speaking | New speaking | Change |
| --- | ---: | ---: | ---: |
| Whole face | 2.786 px | 2.051 px | 26.4% lower |
| Brows | 4.360 px | 3.097 px | 29.0% lower |
| Eyes | 1.470 px | 1.632 px | 11.0% higher |
| Nose | 2.327 px | 1.151 px | 50.5% lower |
| Mouth | 2.660 px | 1.918 px | 27.9% lower |
| Jaw | 2.673 px | 2.066 px | 22.7% lower |

New speaking body offset versus neutral at the first frame:

- Nose center: `(0, -6.25)` px
- Shoulder center: `(1.04, -4.17)` px
- Shoulder width: `-2.06` px

The new speaking asset's own first-to-last whole-face difference is `1.855 px`, compared with `1.421 px` for the previous speaking asset. Therefore the new asset improves cross-pose geometry but slightly worsens its own endpoint consistency.

## WebRTC results

Capture:

`/workspace/MuseTalk/generated/webrtc_pose_showcase/2026-07-30/speaking_smooth_v3_30playback_15lipsync/indian_tutor_speaking_smooth_v3_30playback_15lipsync_webrtc_capture.mp4`

Result:

`/workspace/MuseTalk/generated/webrtc_pose_showcase/2026-07-30/speaking_smooth_v3_30playback_15lipsync/webrtc_result.json`

- H.264 720×1280 at 30 fps
- 5,381 received video frames
- 8,499 received audio frames
- 12/12 ordered idle transitions completed
- 2/2 TTS pose flows completed
- Final pose returned to neutral
- Final stream inactive
- All six avatar caches remained ready

Apples-to-apples comparison against the previous two-frame-crossfade showcase:

| Speaking transition metric | Previous | New | Change |
| --- | ---: | ---: | ---: |
| Mean maximum whole-face RMSE over 400 ms | 3.446 px | 1.949 px | 43.5% lower |
| Worst maximum whole-face RMSE over 400 ms | 6.894 px | 2.818 px | 59.1% lower |
| Mean maximum consecutive-frame RMSE | 1.761 px | 1.333 px | 24.3% lower |
| Worst maximum consecutive-frame RMSE | 2.747 px | 1.484 px | 46.0% lower |

Largest incoming improvements:

- `light_smile → speaking_direct`: `6.894 → 1.934 px` (`71.9%` lower)
- `active_listening → speaking_direct`: `5.857 → 2.818 px` (`51.9%` lower)
- `neutral_resting → speaking_direct`: `3.528 → 2.124 px` (`39.8%` lower)

Residual outgoing changes:

- `speaking_direct → neutral_resting`: `1.377 → 1.354 px`
- `speaking_direct → active_listening`: `1.556 → 1.653 px`
- `speaking_direct → light_smile`: `1.463 → 1.809 px`

## Supporting artifacts

- Geometry JSON: `/workspace/MuseTalk/generated/asset_analysis/2026-07-30_smooth_speaking_v3/geometry_validation.json`
- WebRTC comparison JSON: `/workspace/MuseTalk/generated/asset_analysis/2026-07-30_smooth_speaking_v3/webrtc_transition_comparison.json`
- First-frame contact sheet: `/workspace/MuseTalk/generated/asset_analysis/2026-07-30_smooth_speaking_v3/first_frames_current_and_previous.jpg`
- Previous transition contact sheet: `/workspace/MuseTalk/generated/asset_analysis/2026-07-30_smooth_speaking_v3/previous_20fps_speaking_transition_pairs.jpg`
- New transition contact sheet: `/workspace/MuseTalk/generated/asset_analysis/2026-07-30_smooth_speaking_v3/new_30playback_15lipsync_speaking_transition_pairs.jpg`
