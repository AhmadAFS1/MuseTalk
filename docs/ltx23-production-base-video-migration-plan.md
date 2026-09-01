# LTX 2.3 full-shoulder base-video migration

Date: 2026-09-01 UTC

> Status update: retained inactive alternative. After live review, the user preferred the prior close-up composition. Nothing in this bank was deleted; active production migration moved to `sample_ai_human_ltx23_facetime_closeup_production_v1`. See `docs/ltx23-closeup-production-migration.md`.

## Outcome

MuseTalk now has a new, non-destructive default pose bank for the same sample AI human, framed farther from the camera so both complete shoulders remain inside the image. The previous close-framed LTX assets, ARDY/Kling assets, manifests, and prepared avatar directories are retained unchanged for rollback.

The new immutable pose-set ID is:

```text
sample_ai_human_ltx23_facetime_wide_production_v1
```

Production package:

```text
assets/ltx23_pose_banks/sample_ai_human_facetime_wide_production_v1/
```

Runtime manifest:

```text
configs/pose_test/sample_ai_human_ltx23_facetime_wide_production_v1.json
```

## Physical inventory

Neutral and active listening deliberately alias one physical cache. Direct speech remains one semantic pose but owns two physical variants, so the six logical poses require six unique prepared videos.

| Logical use | File | Frames | Seconds | SHA-256 |
|---|---|---:|---:|---|
| neutral + listening | `idle_active_listening.mp4` | 109 | 4.541667 | `410600a708850a89f4cc21649b47c30b6b443a8c8212b41c14657290dac15ddc` |
| direct V14 | `speaking_direct_v14_subtle.mp4` | 289 | 12.041667 | `62f3169fe703de59b1ff5f8f58808cf17fb4c63dc8f3b536c97e7995a946a476` |
| direct V15 | `speaking_direct_v15_reference_paced.mp4` | 289 | 12.041667 | `69bee20f79f56ee4c0b19aaad8b736cac8cd115f05c439e3e338fe0f16d7f367` |
| nod | `nod_agree.mp4` | 73 | 3.041667 | `5d8e8ba7e9fb9971c1962f67cf25c651dd727735f3f0a0a5b7672998384147cb` |
| empathy | `empathetic_head_tilt.mp4` | 146 | 6.083333 | `d61d346bc61cb29128894432c1c17e49291bb016495ad4d67ffbc7cadc58adb9` |
| smile | `light_smile.mp4` | 145 | 6.041667 | `f89223d14fe66c980c53e2b302ff071b44c1353bb510c704d4a9a0512c5c760b` |

Every file is 480×832, 24 fps, H.264/yuv420p, and silent. The first six and final six decoded RGB frames of every file are identical. All assets share boundary hash:

```text
df2b493201dbbff467a53fd32286c00a28173c1a89290336447abb40c558d9bd
```

This proves all 30 ordered transitions across the six physical videos share the same exact anchor.

## Generation and curation decisions

The wider portrait preserves the original identity, room, lighting, shirt, gaze, and vertical framing while reducing subject scale enough to expose both full shoulders. Its SHA-256 is `de878766585fdace1e8d2d26eda3c02f1fdeb1ea3eae227d53017543a598e7de`.

The successful prompt profiles were reused without rewriting their behavior text:

- V14 and V15 direct speech are direct LTX rerenders from their accepted prompt packs.
- Light smile is a direct rerender of the accepted V8 smile prompt.
- Nod uses the V5 reverse-engineered compact yes-nod prompt because the historical V6 “nod” had actually been a role-promoted source clip rather than a new text generation.
- Listening uses the exact accepted V6 listening prompt. LTX exceeded its two-to-three-degree request in two full stochastic rerolls, so only the restrained opening and settled return regions were retained and recertified.
- Empathy follows the same motion-remaster approach that produced the approved V6 empathy clip. The direct rerender became an off-camera yaw; the production file uses clean near-frontal frames from the new LTX motion trajectory, slowed into one small diagonal out-and-back response.

All rejected and alternate candidates remain under `/workspace/LTX-2.3/musetalk_pose_banks`. Nothing was deleted.

## Runtime migration

The wire contract still exposes exactly these semantic IDs:

1. `neutral_resting`
2. `active_listening`
3. `speaking_direct`
4. `nod_agree`
5. `empathetic_head_tilt`
6. `light_smile`

The runtime adds optional physical variants only beneath `speaking_direct`:

- `v14_subtle` is the top-level fallback.
- `v15_reference_paced` is the alternate.
- `deterministic_boundary_rotation` alternates variants on new assistant turns.
- A retry with the same `turn_id` reuses the same physical render key.
- Semantic traces remain `speaking_direct`; physical traces report a separate render key.

The worker advertises `features.pose_variants_v1=true`. Workers or clients that do not use variants continue to receive the top-level V14 avatar ID.

Default pointers changed in:

- `scripts/test_pose_webrtc.py`
- `templates/webrtc_pose_lab.py`
- `templates/webrtc_wall.py` indirectly through the pose-lab default

No old manifest was edited to point at new bytes. New prepared IDs contain the first eight characters of each content hash, preventing stale-cache collisions.

## Validation status

Completed locally:

- full decode of all six physical MP4s;
- exact geometry, frame-rate, pixel-format, frame-count, and no-audio checks;
- 12 canonical handle checks for every clip;
- all 30 ordered cross-asset boundary comparisons;
- source-to-package byte-identity checks;
- protocol, router, runtime, lab, A/V, crossfade, and wall unit tests;
- Python compilation and JSON validation.

Machine-readable evidence:

```text
assets/ltx23_pose_banks/sample_ai_human_facetime_wide_production_v1/validation_report.json
```

Live migration proof completed after the GPU server restart:

- all six new content-hashed caches were prepared without `--force-recreate`;
- all six new caches load and the prior prepared cache directories remain on disk;
- a two-turn direct WebRTC session passed with the exact configured V14-to-V15 physical rotation;
- a complete six-pose WebRTC cycle passed in the order listening, direct, nod, empathy, smile, neutral;
- both final receiver captures fully decode as 480×832 H.264 video with AAC audio;
- timestamp-locked receiver validation passed, including the permitted one-normal-audio-packet jitter-buffer observation after a declared rebase;
- the focused protocol/router/runtime/lab/A/V/crossfade/wall suite passed all 83 tests.

Final human visual acceptance of the wider framing, lip sync, blink timing, and motion amplitude remains intentionally pending. That review does not block rollback or cache integrity.

Proof artifacts:

```text
generated/webrtc_pose_showcase/2026-09-01/ltx23_full_shoulders_production_v1/
```

## Rollback

Rollback is a pointer change for new sessions: restore the previous pose-set manifest as the lab/application default. Existing calls remain pinned to the pose set with which they started. The previous source videos and prepared avatar directories are intentionally retained; deletion is outside this migration and requires a separate explicit cleanup request.

The Lingua repository is not present on this host. Its character configuration must be updated separately to request `sample_ai_human_ltx23_facetime_wide_production_v1` when that application is available.
