# ARDY v4 → Segmind → MuseTalk WebRTC validation

Date: 2026-07-30

## Outcome

The v4 batch is the smoothest aggregate four-pose WebRTC session tested so far under the same 30 fps playback, 15 fps MuseTalk lip-sync, and two-frame crossfade profile.

It is not better on every individual edge. Segmind regenerated `light_smile` with a larger eyebrow-state difference, so `light_smile → neutral_resting` and `light_smile → speaking_direct` are local regressions.

## ARDY v4 source validation

Delivery:

`/workspace/ardy/outputs/pose_delivery_mvp_four_v4_30frame_boundaries`

- Four independent videos with 30-frame opening and ending handles.
- 12/12 directed transitions pass.
- 4/4 self-loops pass.
- All 360 aligned handle-frame comparisons are pixel-identical.
- Last outgoing frame → first incoming frame: MAE `0`, RMSE `0`, SSIM `1.000`.
- Shared decoded handle SHA-256: `a5396c0f38dd36c5593839edfb1a020c6b57445f540ff1b1b8051b5a15569a2b`.
- The certified v3 delivery remains preserved.

## Segmind generation

Exactly four requests were submitted, with zero retries.

| Pose | Request ID | Cost | Output frames |
| --- | --- | ---: | ---: |
| Neutral | `e68dc44c55b88fd25a1ff189b5510b9e` | $1.5302 | 328 |
| Speaking | `ffe6d9ee931c9dbea60af82b3cf6eb13` | $1.68 | 360 |
| Smile | `9c7312acb149ef98fa72fe6211830f3b` | $0.84 | 180 |
| Listener/empathy | `dc848d920339591ce822a65780e6b193` | $1.40 | 300 |

Total billed inference cost: **$5.4502**.

The same Indian-tutor portrait and established prompt template were used. Identity, framing, direct gaze, closed mouth, and absence of persistent squint passed visual sampling.

Segmind returned neutral as 328 frames / 10.933 seconds instead of the 330-frame / 11-second reference. The runtime manifest and prepared cache record the returned media facts.

## Segmind boundary geometry

Segmind still does not preserve the exact ARDY pixels:

- Exact first-frame pose pairs: `0/6`.
- Exact directed last-frame → first-frame transitions: `0/12`.
- Mean static directed-boundary whole-face RMSE changed from `1.848 px` in v3 to `1.890 px` in v4.
- Worst static directed-boundary RMSE changed from `2.264 px` to `2.761 px`, driven by the regenerated smile.

First-frame whole-face difference versus neutral:

| Pose | v3 | v4 | Result |
| --- | ---: | ---: | --- |
| Speaking | 2.051 px | 1.550 px | Improved 24.4% |
| Listener | 2.233 px | 2.277 px | Approximately unchanged |
| Smile | 1.796 px | 2.603 px | Worsened 44.9% |

The v4 smile eyebrow difference versus neutral is `4.832 px`, compared with `2.318 px` in v3.

Body alignment improved: v4 shoulder-center offsets are at most `3.125 px` vertically, with essentially zero shoulder-width drift.

## MuseTalk installation and WebRTC test

All installed runtime videos and their prepared `input_video.mp4` files have matching SHA-256 hashes. The previous v3 runtime assets are preserved under:

`/workspace/MuseTalk/generated/asset_analysis/2026-07-30_mvp_four_v4_30frame_boundaries/previous_runtime_assets`

WebRTC capture:

`/workspace/MuseTalk/generated/webrtc_pose_showcase/2026-07-30/mvp_four_v4_30frame_boundaries_30playback_15lipsync/indian_tutor_mvp_four_v4_30frame_30playback_15lipsync_webrtc_capture.mp4`

Test result:

- `ok=true`
- H.264 720×1280 at 30 fps
- 7,477 video frames / 249.233 seconds
- 11,164 audio frames
- 12/12 ordered idle transitions completed
- Empathy and warmth TTS flows completed
- 15 fps MuseTalk generation, including 180 speaking frames for the 12-second speaking segment
- Final pose: neutral
- Final stream active: false

## V4 versus v3 smoothness

Both recordings use 30 fps playback, 15 fps MuseTalk lip-sync, and a two-frame / 66.7 ms crossfade.

| Metric | v3 | v4 | Change |
| --- | ---: | ---: | ---: |
| All transitions: mean 100 ms whole-face RMSE | 2.206 px | 2.031 px | 7.9% lower |
| All transitions: mean 400 ms whole-face RMSE | 2.473 px | 2.131 px | 13.8% lower |
| All transitions: worst 400 ms whole-face RMSE | 5.059 px | 3.953 px | 21.9% lower |
| Speaking idle edges: mean 400 ms RMSE | 1.949 px | 1.755 px | 10.0% lower |
| Speaking idle edges: worst 400 ms RMSE | 2.818 px | 2.581 px | 8.4% lower |
| TTS edges: mean 400 ms whole-face RMSE | 4.296 px | 3.281 px | 23.6% lower |

Strong improvements:

- `neutral_resting → speaking_direct`: `2.124 → 1.386 px`, 34.8% lower.
- `active_listening → speaking_direct`: `2.818 → 2.050 px`, 27.2% lower.
- `light_smile → active_listening`: `1.578 → 1.263 px`, 20.0% lower.
- `neutral_resting → light_smile`: `2.569 → 2.039 px`, 20.7% lower.

Local regressions:

- `light_smile → neutral_resting`: `1.399 → 2.207 px`, 57.8% higher.
- `light_smile → speaking_direct`: `1.934 → 2.581 px`, 33.4% higher.
- `active_listening → neutral_resting`: `1.848 → 1.911 px`, 3.4% higher.

## Verdict

Use v4 when the priority is overall call smoothness, speaking transitions, and TTS transitions. Keep the two-frame crossfade.

V4 does **not** earn an “every transition improved” result because of the new Segmind smile geometry. If every directed edge must beat v3, `light_smile` is the only asset that should be addressed next; the v4 neutral, listener/empathy, and speaking outputs should be retained.

## Evidence

- WebRTC result: `/workspace/MuseTalk/generated/webrtc_pose_showcase/2026-07-30/mvp_four_v4_30frame_boundaries_30playback_15lipsync/webrtc_result.json`
- Cadence-neutral comparison: `/workspace/MuseTalk/generated/asset_analysis/2026-07-30_mvp_four_v4_30frame_boundaries/v4_vs_v3_cadence_neutral_webrtc_validation.json`
- Segmind geometry: `/workspace/MuseTalk/generated/asset_analysis/2026-07-30_mvp_four_v4_30frame_boundaries/v4_segmind_geometry_validation.json`
- Transition contact sheet: `/workspace/MuseTalk/generated/asset_analysis/2026-07-30_mvp_four_v4_30frame_boundaries/v3_vs_v4_idle_transition_pairs_400ms.jpg`
- First-frame comparison: `/workspace/MuseTalk/generated/asset_analysis/2026-07-30_mvp_four_v4_30frame_boundaries/v4_and_v3_first_frames.jpg`
