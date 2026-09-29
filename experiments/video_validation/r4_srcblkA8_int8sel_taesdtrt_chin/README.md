# Round r4 — 400+ fps with 100% chin, layer-selective INT8 (2026-09-28)

**Result:** **414.9 fps** aggregate (415.6 / 414.1 in two 62 s timed runs, 25,920 frames each). This uses the same
six-stream full recipe as r2/r3: TAESD + native encoder + 100% chin + refined seam, with the chin code unchanged
and every frame tracked.

**What changed vs r2 (350.2 fps).** INT8 Q/DQ is applied to 146 chosen layers (59% of UNet MACs) in `down1`, `down2`,
`down3`, `mid`, `up0`, `up1` and `up2`, instead of FP16.
- The recipe `blkA_thr_8e-06` comes from the per-layer fake-quant study in
  `docs/fps_comparisons/4070s_400fps_20260928/`.
- It keeps FP16 wherever INT8 hurt most: all of `down0` and `up3`, the audio cross-attention K/V projections, and
  the most sensitive single layers.
- Activation ranges come from max or MSE calibration per layer. There is no retraining.
- Engine set: `models/tensorrt_unet_stagewise_sm89_srcblkA8/bs16`. It is srcmix's prefix, `down0rest`, `up3` and
  tail, plus the `blkA_thr_8e-06` blocks.

**How it compares with r3 (415.6 fps, broad INT8):** the same speed, with about 6× lower latent error and about
half the pixel-level change.

| Metric (6 identities, raw pre-encode frames vs accepted render) | r4 (414.9 fps) | r3 (415.6 fps) | r2 (350.2 fps) | Synthetic ±1 LSB noise |
|---|---|---|---|---|
| UNet latent mae_max / max_abs, main (gate 0.01 / 0.5) | 0.0071 / 1.28 (mae pass, max_abs fail) | 0.038 / 2.39 | 0.0025 / 0.39 | — |
| Lip-aperture correlation | 0.9989-0.9994 | 0.9969-0.9986 | 0.9998-1.0000 | 0.9999-1.0000 |
| Aperture mean abs delta (px) | 0.13-0.24 | 0.23-0.37 | 0.04-0.08 | 0.04-0.07 |
| Mouth flicker ratio | 0.998-1.003 | 1.017-1.025 | 0.9996-1.0000 | 1.003-1.008 |
| Jaw flicker ratio | 0.998-1.002 | 0.999-1.009 | 0.9998-1.0000 | 1.000-1.003 |
| Jaw+lip landmark deviation mean (px) | 0.105-0.171 | 0.197-0.355 | 0.036-0.094 | 0.045-0.088 |
| Same, p99 (px) | 0.37-0.65 | 0.65-1.01 | 0.11-0.47 | 0.13-0.33 |
| Chin-target error change (px, limit +0.05) | -0.015 to +0.024 | -0.010 to +0.035 | -0.005 to +0.032 | -0.012 to +0.013 |
| Face / mouth PSNR (dB) | 54.9-57.6 / 47.7-50.1 | 46.9-51.8 / 40.2-44.2 | 60.8-64.2 / 55.1-57.7 | 56.9-57.8 / 51.5-51.9 |
| Mouth sharpness ratio | 0.997-1.005 | pass | 1.000-1.001 | — |

- **Gates.** Every perceptual gate passes on 6/6: lip correlation and lag, aperture delta, flicker, chin-target
  error, protected lips and sharpness. The landmark gate fails both the strict bar (0.05 / 0.15 px) and the
  calibrated bar (0.10 / 0.35 px) on 6/6.
- **Flicker.** Unlike r3, r4 adds no temporal flicker (ratios ≈ 1.000).
- **Error type.** The remaining change is a small systematic shift. The aperture delta is about 2-3× what ±1 LSB of
  random noise causes, at a similar PSNR.

**Videos:** `<identity>_ab.mp4` for all six identities, plus `mosaic_candidate.mp4`. The layout is the same as r2/r3:
- A (left) is the accepted pre-change render; B (right) is this candidate.
- The bottom row has a 3× mouth zoom and the exact raw 256 px face |A-B| ×8 panel.
- The metrics line comes from `scripts/quality_ab_metrics.py`.

**Next step:** INT8 quality recovery at the same speed. That means learned activation ranges and per-channel bias
corrections, trained against the FP16 output (`scripts/int8_layer_study.py --stage recover`). See
`docs/fps_comparisons/4070s_400fps_20260928/README.md` §8.
