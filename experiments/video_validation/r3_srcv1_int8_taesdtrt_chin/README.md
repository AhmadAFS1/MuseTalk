# Round r3 — 400 fps attempt with 100% chin (2026-09-28)

**Result:** **415.6 fps** aggregate (416.6 / 414.7 in two 62 s timed runs, 25,920 frames each), using the same
six-stream full recipe as r2: TAESD + native encoder + 100% chin + refined seam, chin code unchanged.

**What changed vs r2:** INT8 Q/DQ on six UNet blocks (`down1`, `down2`, `down3`, `mid`, `up2`, `up3`), using plain
post-training quantization (modelopt max calibration) and no quality recovery. Engine set:
`models/tensorrt_unet_stagewise_sm89_srcv1/bs16`.

**This is a quality trade, not FP16 noise.**

| Metric | r3 (415.6 fps) | r2 (350.2 fps) | Reference: accepted INT8-VAE vs TAESD decoder switch |
|---|---|---|---|
| UNet latent mae_max / max_abs | 0.038 / 2.39 (**fails** the 0.01 / 0.5 gate) | 0.0025 / 0.39 | — |
| Raw face diff mean / max | 1.3 LSB / ~99 LSB | 0.1 LSB / ~29 LSB | — |
| Lip-aperture correlation | 0.997-0.998 | 0.9998+ | 0.992-0.993 |
| Aperture mean abs delta | 0.28-0.33 px | ~0.05-0.07 px | 0.62-0.65 px |
| Mouth flicker ratio | 1.017-1.021 | ~1.000 | 0.92-0.95 |
| Landmark deviation mean / p99 | 0.20-0.36 / 0.65-0.98 px (fails both bars) | 0.04-0.09 / 0.11-0.47 px | 0.47-0.50 / 1.38-1.41 px |
| Chin-target error change / sharpness | pass / pass | pass / pass | — |

The measured differences are about half the size of the INT8-VAE vs TAESD decoder switch, which you judged
"essentially the same". The exact diff panel still shows real structure around the lips, teeth and beard. Watch for
temporal shimmer at normal speed. If it isn't acceptable, the next step toward 400 is quality recovery of the INT8
blocks (per-layer FP16 carve-outs, then scale-only QAT / distillation), not a different recipe.
