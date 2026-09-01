# LTX 2.3 close-up production migration

Date: 2026-09-01 UTC

## Decision

The user preferred the original close-up composition because the full-shoulder character occupied too little of a vertical video-call frame. MuseTalk therefore uses the close-up bank as the new-session default. The full-shoulder bank is retained unchanged as an inactive alternative.

Active pose-set ID:

```text
sample_ai_human_ltx23_facetime_closeup_production_v1
```

Runtime manifest:

```text
configs/pose_test/sample_ai_human_ltx23_facetime_closeup_production_v1.json
```

Immutable asset package:

```text
assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/
```

## Approved asset selection

| Use | Approved source | SHA-256 | Existing prepared avatar ID |
|---|---|---|---|
| neutral + listening | V6 active listening | `099877cef231ce12dede03843c558d10c2fa1e9c4e054c83be595e81a00f6ae4` | `sample_ai_human_ltx23_user_review_six_v6_active_listening` |
| direct V14 | gaze-locked subtle motion | `20f98455436b4f6f77525edc89a2c89495b67c15b710cd1332ec0e783e6670e0` | `sample_ai_human_ltx23_v14_speaking_comparison_v1` |
| direct V15 | reference-paced motion | `c8ce2774f916a7466e7ff28fb25583ab617bc01f744069c90e997dbf45605db0` | `sample_ai_human_ltx23_v15_speaking_comparison_v1` |
| nod | V6 approved nod | `96b71f04ccfec8750f8534c2b9e8d3c2064c492cb9fd1e89e69d0f6505b5b604` | `sample_ai_human_ltx23_user_review_six_v6_nod_agree` |
| empathy | V6 slow tilt | `d8260d60bdd4288150502c214eed1f9587dfb493afbd391471a9c4441e039327` | `sample_ai_human_ltx23_user_review_six_v6_empathetic_head_tilt` |
| smile | V8 moderate smile | `55561a4f0bb7e5e7ea1a5598ceb6e36b5f579239958fd25a27bccda3bc1a2ed6` | `sample_ai_human_ltx23_user_review_six_v8_light_smile` |

Neutral and active listening intentionally share one physical file. Direct speech remains one semantic pose with deterministic V14/V15 rotation on new assistant turns. Retrying the same `turn_id` keeps the same physical variant.

## Integrity

Every packaged video is byte-identical to its certified LTX source and its existing MuseTalk `input_video.mp4` cache. This was checked with SHA-256 and `cmp`; no cache was recreated.

All six physical videos are 480×832, 24 fps, H.264/yuv420p, and silent. Their first six and final six decoded frames share RGB hash:

```text
47b05c6bdd63466e13381dc6cf21545e827bea0bc668c5798cbf7c69f7076b33
```

Full decode, exact boundary certification, and all 30 directed cross-asset transitions passed. Machine evidence is in the bank's `validation_report.json`.

## Runtime switch

The default manifest pointers in the worker pose lab, wall, and headless WebRTC harness now select the close-up production manifest. Existing sessions remain pinned to the pose set supplied when they were created; the default change applies to new sessions.

The wider bank remains available at:

```text
assets/ltx23_pose_banks/sample_ai_human_facetime_wide_production_v1/
configs/pose_test/sample_ai_human_ltx23_facetime_wide_production_v1.json
```

No source video, generated candidate, manifest, prepared cache, or WebRTC proof was deleted.

## Verification status

Migration is complete and live-verified:

- the server restarted with the close-up manifest embedded in `/webrtc/pose-lab`;
- all six existing prepared caches warmed with `cached=true` and `disk_prepared=true`;
- no cache preparation or force recreation occurred;
- WebRTC session `y86vQ74o-7VG8dZkGeKC4Q` completed active listening at 0.381 s, V14 direct at 10.308 s, nod at 22.606 s, empathy at 28.679 s, smile at 38.822 s, and neutral at 45.063 s;
- the same session then selected V14 for speaking turn one and V15 for speaking turn two;
- receiver timestamp validation and configured-versus-observed variant rotation both passed;
- the 79.474-second 480×832 H.264/AAC recording fully decodes;
- the complete focused regression suite passed all 83 tests.

Proof capture:

```text
generated/webrtc_pose_showcase/2026-09-01/ltx23_closeup_production_v1/sample_ai_human_ltx23_closeup_production_v1_webrtc_capture.mp4
```
