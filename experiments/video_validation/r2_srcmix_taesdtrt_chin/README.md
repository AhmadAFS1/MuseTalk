# Round r2 — 350 fps with 100% chin (2026-09-28)

**Result:** **350.2 fps** aggregate (350.7 / 349.7 in two 66 s timed runs, 23,040 frames each). Six concurrent
streams run the full required recipe: TAESD decoder + native avatar encoder + 100% chin alignment + refined seam,
with the chin code (`character_factory/h3_avatar_workflow/chin.py`) run unchanged and every frame tracked.
The before-point is the accepted single-stream render loop, at 148-171 fps per identity (`render.json`).

**Configuration (candidate B):**

| Component | Value |
|---|---|
| UNet | stagewise FP16 bs16 TensorRT, with a source-prefix cache (conv_in + down0.resnets[0] computed once per source frame, `forward_cached` bit-identical to `forward`) and INT8 Q/DQ on `down3` + `mid` only |
| Decoder | TensorRT TAESD (full height, fused uint8 pack), strict (no fallback) |
| Engine set | `models/tensorrt_unet_stagewise_sm89_srcmix/bs16` |
| Harness | `scripts/chin_multistream_render.py --backend stagewise16_taesdtrt --flags MUSETALK_UNET_STAGEWISE_CACHE_DIR=models/tensorrt_unet_stagewise_sm89_srcmix`, one ordered worker per stream |

**Videos:** `<identity>_ab.mp4` for all six identities, plus `mosaic_candidate.mp4`.
- A (left) is the accepted pre-change render; B (right) is the candidate. Both play at the render's native 24 fps with the same audio.
- The bottom row is a 3x mouth zoom, plus the exact raw 256 px generated-face |A-B| x8 panel (black means identical).
- The metrics line comes from `scripts/quality_ab_metrics.py`.

## Quality (raw pre-encode frames, candidate vs accepted, all 6 identities)

| Gate | Result |
|---|---|
| Lip-aperture correlation / lag | 0.9998-1.0000 / 0 frames: PASS 6/6 |
| Flicker ratio (mouth, jaw, seam ring) | 0.9996-1.0004: PASS 6/6 |
| Protected lips vs own standard compose | 0 change: PASS 6/6 |
| Chin-target error change | -0.005 to +0.032 px (limit +0.05): PASS 6/6 |
| Mouth sharpness ratio | 1.000-1.001: PASS 6/6 |
| Jaw+lip landmark deviation, strict 0.05 / 0.15 px | PASS 2/6 (0.036-0.094 mean, 0.110-0.475 p99) |
| Same, calibrated 0.10 / 0.35 px (report-only proposal) | PASS 5/6; the full-beard identity is at p99 0.475 |

The strict landmark gate sits at FaceMesh's own sensitivity floor: ±1 LSB of random noise fails it on 3-5 of the 6
identities (`docs/fps_comparisons/4070s_300fps_impl_20260928/quality_metrics/calibration_summary.md`). The overlay
therefore shows `verdict FAIL` for those identities, and every perceptual gate passes. **Your visual verdict decides.**

UNet quality gate (`validate_unet_backend.py`, 14-avatar corpus): main mae_max 0.00249 / max_abs 0.389; holdout
0.00213 / 0.242. The limits are 0.01 / 0.5; the shipping `.ts` engine scores 0.0026 / 0.402.
