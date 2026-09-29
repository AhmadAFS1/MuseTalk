# Lineage videos, BEFORE/r2/r3/r4 (2026-09-28): SUPERSEDED, use `../lineage_all_rounds/`

This first version was reviewed by six per-identity verifier agents and a skeptic. They confirmed it is bit-exact
and correct, and raised presentation issues that the newer sets fix:
- no gate verdicts were shown;
- BEFORE's fps was single-stream only (like-for-like: 252 fps);
- the diff row covered the raw generated face rather than the composited output, and 1-LSB changes vanished in the
  encode;
- the "3x" zoom label was inaccurate;
- dividers crossed the title.

Here "r4 NEW" means new at the time; r5 came later. Made with an earlier `scripts/video_lineage.py`: 400 px columns,
crf 14, arms r2/r3/r4.

Each identity gets one video that puts the working pre-change render and the optimization rounds side by side, with
the same frames, audio and chin recipe. Made with `scripts/video_lineage.py`.

| Column | What it is | Measured throughput |
|---|---|---|
| **BEFORE** | The accepted render (`/workspace/experiments/avatar_diversity_20260927/<id>/`). Shipping TensorRT FP16 bs8 UNet + compiled TAESD + 100% chin + refined seam. | single stream, 148–171 fps (`render.json`) |
| **r2** | Stagewise FP16 bs16 UNet + source-prefix cache + INT8 `down3`/`mid` + TensorRT TAESD | 350.2 fps aggregate, 6 streams |
| **r3** | r2 + broad INT8 PTQ on 6 UNet blocks (no recovery) | 415.6 fps aggregate |
| **r4 (new)** | r2 + layer-selective INT8 (146 layers; `down0`, `up3` and the audio K/V projections stay FP16) | 414.9 fps aggregate |

## Videos

- [black_man_short_beard_lineage.mp4](black_man_short_beard_lineage.mp4)
- [black_woman_lineage.mp4](black_woman_lineage.mp4)
- [east_asian_man_goatee_lineage.mp4](east_asian_man_goatee_lineage.mp4)
- [middle_eastern_man_full_beard_lineage.mp4](middle_eastern_man_full_beard_lineage.mp4)
- [south_asian_woman_lineage.mp4](south_asian_woman_lineage.mp4)
- [white_man_clean_shaven_lineage.mp4](white_man_clean_shaven_lineage.mp4)

Each video is 1600×1612 at 24 fps, 10 s, with the identity's speech. Each identity also has `<id>_stills.png`, two
frames at half size:
- the widest mouth opening;
- the frame where r4 differs most from BEFORE.

**How to read a video.**
- Top row: the full frame.
- Middle row: a 3× mouth zoom (nearest neighbour, so pixels are real).
- Bottom row: the raw 256 px generated face in the BEFORE column, and **|face − BEFORE face| × 8** for each round.
  Black means identical. A difference of 1 LSB shows as 8 LSB.
- Cyan lines: the quality tool's numbers for that round vs BEFORE, on raw frames. That is lip-aperture correlation
  and delta, flicker ratios, jaw+lip landmark deviation, PSNR and chin-target error change.

## Why these are fair

- **Raw frames.** Every column is rebuilt from its raw pre-encode frames, not from stored mp4s. The rebuild uses the
  unchanged `chin.py` compose, through the quality tool's `load_arm`.
- **Bit-exact rebuilds.** Every column was checked (`lineage_report.json`):
  - BEFORE matches the accepted render's recorded `raw_refined_sha256` on 6/6 identities;
  - each round matches the `raw_refined_sha256` its multi-stream harness worker recorded while rendering it, on
    18/18.
- **One encode for all.** The whole canvas is encoded once (libx264 crf 14, yuv420p). No column carries an extra
  lossy generation.
- **Why the older A/B videos are biased.** The per-round A/B videos (`r2_*/`, `r3_*/`, `r4_*/`) put the stored
  accepted `refined_raw.mp4` (~1.9 Mbit/s) next to crf-12 captures (~5.3 Mbit/s). That gives the candidate a slight
  compression advantage. Use these lineage videos for visual judgement.

## What the numbers say (all 6 identities)

| Round | fps | Lip corr | Aperture delta (px) | Mouth flicker | Landmark mean / p99 (px) | Mouth PSNR (dB) |
|---|---|---|---|---|---|---|
| r2 | 350.2 | ≥ 0.9998 | 0.04–0.08 | ~1.000 | 0.036–0.094 / 0.11–0.47 | 55.1–57.7 |
| r3 | 415.6 | 0.9969–0.9986 | 0.23–0.37 | 1.017–1.025 | 0.20–0.36 / 0.65–1.01 | 40.2–44.2 |
| r4 | 414.9 | 0.9989–0.9994 | 0.13–0.24 | 0.998–1.003 | 0.105–0.171 / 0.37–0.65 | 47.7–50.1 |

For scale: ±1 LSB of random noise on the faces gives landmarks 0.045–0.088 / 0.13–0.33 px and mouth PSNR ~51.5 dB.

Regenerate with:

```
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/video_lineage.py
```

Add a round with repeated `--arm 'name|fps|what|capture_dir|quality_label'` options. They replace the defaults, so
list every round you want.
