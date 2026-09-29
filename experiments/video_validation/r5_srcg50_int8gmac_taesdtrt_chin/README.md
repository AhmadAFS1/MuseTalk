# Round r5: ~400 fps with 100% chin, INT8 layers chosen by error per MAC (2026-09-28/29)

**Throughput.** Six-stream full recipe as r2–r4: TRT TAESD + native encoder + 100% chin + refined seam, with the
chin code unchanged. 25,920 frames per 64 s repeat.

| Run | Repeats (fps) |
|---|---|
| `T_srcg50` | 401.8 / 400.1 (median 400.9) |
| `T_srcg50_pair` (back to back with BEFORE) | 401.4 / 400.1 |
| `T_srcg50_sustained` (5 consecutive repeats) | 404.0 → 400.9 → 400.5 → 400.1 → 399.96 |

- **At the line, no margin.** In the sustained run the GPU warms from 64 to 67 °C and the SM clock drops from 2475
  to 2460 MHz under the 220 W cap, so r5 sustains ≈400.0 fps. Longer runs were not measured.
- **Like-for-like: 1.59× the previous working pipeline.** The pre-change backends in the same harness, run back to
  back, give 252.6 / 251.4 fps, with every clip bit-identical to the accepted-recipe renders. The single-stream
  148–171 fps render loop is not a like-for-like baseline.

**What changed vs r2 (350.2 fps).**
- **INT8 layers:** 117 UNet layers in total, i.e. 50% of MACs and 99 layers beyond r2's `down3`/`mid`, across
  `down1`, `down2`, `down3`, `mid`, `up0`, `up1` and `up2`.
- **How they were chosen:** greedily, by the single-layer INT8 error each adds per MAC
  (`docs/fps_comparisons/4070s_400fps_20260928/`, §8.4–8.5).
- **Kept in FP16:** `down0`, `up3`, the audio cross-attention K/V projections and `up2.upsamplers.0.conv`.
- **Relation to r4:** it is a strict subset of r4's 146 layers.
- **Engine set:** `models/tensorrt_unet_stagewise_sm89_srcg50/bs16`.
- **Selection:** under the 400 fps budget, the all-`gmac_0.50` combination tied for the lowest predicted error
  (`select_combo.py`).

**Videos.** Every column is rebuilt from raw frames, bit-exact to what was rendered, and encoded once. This folder
holds no videos of its own.
- [`../focus_before_r2_r5/`](../focus_before_r2_r5/README.md): BEFORE | r2 | r5 at native 512 px.
- [`../lineage_all_rounds/`](../lineage_all_rounds/README.md): BEFORE | r2 | r3 | r4 | r5.

## Quality (raw pre-encode frames vs BEFORE, 6/6 identities)

| Metric | r5 (≈400 fps) | r4 (414.9) | r2 (350.2) | ±1 LSB random noise |
|---|---|---|---|---|
| UNet latent mae_max / max_abs, main (gate 0.01 / 0.5) | 0.0044 / 0.78 (max_abs FAIL) | 0.0071 / 1.28 | 0.0025 / 0.39 | — |
| UNet latent, holdout | 0.0039 / 1.26 | 0.0061 / 1.80 | 0.0021 / 0.24 | — |
| Lip-sync correlation | 0.9997–0.9998 | 0.9989–0.9994 | 0.9998–1.0000 | 0.99985–1.0000 |
| Aperture delta (px) | 0.08–0.14 | 0.13–0.24 | 0.04–0.08 | 0.04–0.07 |
| Mouth / jaw flicker ratio | 0.999–1.001 / 0.999–1.000 | 0.998–1.003 | ~1.000 | 1.003–1.008 |
| Jaw+lip landmark deviation mean / p99 (px) | 0.064–0.126 / 0.20–0.47 | 0.105–0.171 / 0.37–0.65 | 0.036–0.094 / 0.11–0.47 | 0.045–0.088 / 0.13–0.33 |
| Landmark repo gate (0.05 / 0.15) / proposed bar (0.10 / 0.35) | 0/6 / 4/6 | 0/6 / 0/6 | 2/6 / 5/6 | — |
| All other quality-tool gates (lip, flicker, chin, protected lips, sharpness) | 6/6 | 6/6 | 6/6 | — |
| Chin-target error change (px, limit +0.05) | −0.016 to +0.020 | −0.015 to +0.024 | −0.005 to +0.032 | — |
| Face / mouth PSNR, mean (dB) | 57.6–60.7 / 51.0–53.4 | 54.9–57.6 / 47.7–50.1 | 60.8–64.2 / 55.1–57.7 | 56.9–57.8 / 51.5–51.9 |
| Mouth PSNR, worst frame (dB) | 43.1–47.0 | 35.3–39.8 | 46.9–53.9 | 51.4–51.6 |

**Reading the table:**
- **Mean mouth change.** r5's is about the size of ±1 LSB of random noise on the generated face.
- **Worst frames and landmarks.** These are larger:
  - worst frames are 4–8 dB below the noise arm;
  - aperture and landmark changes are about 1.5–2× the noise arm.

  The changes sit on the lip contours, teeth and beard edges, and at normal viewing they are not visible (see the
  videos' diff row).
- **Proposed-bar misses:** the short beard is near the line (0.104 / 0.36). The full beard misses clearly (0.126 /
  0.47), where r2 also misses (p99 0.475).
- **Decoder gate:** r2–r5 share the TensorRT TAESD decoder. Its gate records FAIL on the 3 LSB max bar (max 5 LSB;
  the mean of 0.066 passes). It has been used since r2; your decision on it is still pending.
- **BEFORE:** the renders come from the accepted recipe, but their own visual acceptance is still pending
  (`validation.json`).
