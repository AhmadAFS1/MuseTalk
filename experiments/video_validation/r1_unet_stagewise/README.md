# Round r1 — stagewise FP16 bs16 UNet (2026-09-28)

**What changed:** only the UNet backend. The shipping TensorRT `.ts` FP16 bs8 engine is replaced by
the stagewise FP16 bs16 engine set (`MUSETALK_UNET_BACKEND=trt_stagewise`, `MUSETALK_UNET_STAGEWISE_BATCH=16`).
The TAESD decoder (compiled), native avatar encoder, 100% chin alignment and refined seam are unchanged.
The render loop is the accepted `character_factory/h3_avatar_workflow/render_stage.py`, run unmodified
through a shim that only swaps the UNet loader.

**Columns:** A (left) is the pre-change **accepted** render (`experiments/avatar_diversity_20260927/<id>/refined_raw.mp4`).
B (right) is the candidate. Both play at the render's native 24 fps with the same audio, and both are crf 18 encodes.
The bottom row shows a 3x nearest-neighbour mouth zoom (A | B) and the raw 256 px generated face
`|A-B| x8` taken from `faces.npz`; that panel is exact, with no codec noise, and black means identical.

| Identity | Video |
|---|---|
| South Asian woman | `south_asian_woman_ab.mp4` |
| Middle Eastern man, full beard | `middle_eastern_man_full_beard_ab.mp4` |

## Speed (GPU path, sustained, real multi-avatar inputs)

| | UNet ms / 16 frames | TAESD ms / 16 | Aggregate fps | Run |
|---|---|---|---|---|
| Before: `.ts` bs8 + compiled TAESD | 48.4 | 12.7 | **260.1** | 200 s |
| After: stagewise bs16 + compiled TAESD | 38.2 | 13.2 | **309.6** | 120 s |

These are GPU-path numbers: UNet, TAESD, post-processing and pinned copy to host, with no chin or serving.
The single-stream chin render loop in this A/B stays CPU-bound (~150-170 fps), as it was before.
Multi-stream chin throughput is measured in round r2.

## Quality (A vs B)

| Metric | South Asian woman | Middle Eastern man (beard) | Gate |
|---|---|---|---|
| raw face diff mean / max | 0.06 / 11 LSB | 0.12 / 17 LSB | mean <= 0.2 |
| pixels > 3 LSB | 0.00 % | 0.01 % | report |
| lip-aperture correlation (lip sync) | 1.0000 | 0.9999 | >= 0.97 |
| flicker ratio B/A (lower-face temporal diff) | 1.0002 | 0.9998 | <= 1.05 |
| protected-lip change / reference match / min Jacobian (B) | 0 / exact / 0.73 | 0 / exact / 0.85 | 0 / exact / > 0.25 |
| FaceMesh landmark deviation mean / p99 | 0.036 / 0.093 px | 0.073 / **0.238 px** | 0.05 / 0.15 (strict) |
| max chin-delta difference (any frame) | 0.008 | 0.133 (frame 138) | report |

**Honest note:** for the beard identity, landmark deviation p99 is above the strict FP16-noise bar.
The tracker amplifies ~0.1 LSB face differences in the beard texture. Frame 138, the worst chin-delta
frame, is visually indistinguishable in a 2x chin crop; its |A-B| is diffuse codec noise with no jaw or
beard-edge structure (`beard_worst_chin.jpg`). The user judges it on video.

UNet quality gate on the 14-avatar corpus: main mae_max 0.0025 / max_abs 0.431; holdout 0.0021 / 0.317
(shipping `.ts`: 0.0026 / 0.402). The limits are 0.01 / 0.5.
