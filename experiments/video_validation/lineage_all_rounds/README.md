# Lineage videos: the previous working pipeline vs every optimization round, r2–r5 (2026-09-28/29)

Each identity gets one video with the same frames, audio and 100% chin recipe in every column. Made with
`scripts/video_lineage.py`. For the closest look at the previous working pipeline, the previous round (r2) and the
newest (r5), see [`../focus_before_r2_r5/`](../focus_before_r2_r5/README.md): three columns at 512 px, where every
source pixel is drawn.

| Column | What it is | Aggregate fps, 6 streams, same harness | vs BEFORE |
|---|---|---|---|
| **BEFORE** | Previous working pipeline: shipping TensorRT FP16 bs8 UNet + compiled TAESD + 100% chin + refined seam. The pre-change renders in `/workspace/experiments/avatar_diversity_20260927/<id>/`, made with the accepted recipe; their visual acceptance is still pending (`validation.json`). | **252.0** (single-stream render loop: 148–171) | 1.00× |
| **r2** | Stagewise FP16 bs16 UNet + source-prefix cache + INT8 `down3`/`mid` + TensorRT TAESD | 350.2 | 1.39× |
| **r3** | r2 + broad INT8 PTQ on 6 UNet blocks, no recovery | 415.6 | 1.65× |
| **r4** | r2 + layer-selective INT8 (146 layers; `down0`, `up3` and audio K/V stay FP16) | 414.9 | 1.65× |
| **r5 (new)** | r2 + INT8 chosen by error per MAC (117 layers in total, 99 beyond r2; `gmac_0.50`) | **≈400 sustained** (404.0 → 399.96 over 5 × 64 s) | **1.59×** |

## Where the fps numbers come from

- **BEFORE.** 252.0 fps is the pre-change backends in the same six-stream harness, re-measured back to back with r5
  (`T_baseline_pair` 252.6 / 251.4 vs `T_srcg50_pair` 401.4 / 400.1). Every BEFORE clip is bit-identical to the
  pre-change renders. That baseline pair ran with `--compare-accepted`, about 1 s of CPU per worker, and the GPU was
  99.9% busy in both runs. An earlier run with an older harness revision gave 251.8.
- **r2–r4.** Same harness code, but measured hours apart, not back to back.
- **The single-stream loop.** Its 148–171 fps overstates the gains if used as the baseline.
- **r5.** Consecutive 64 s repeats run 404.0, 400.9, 400.5, 400.1 and 399.96 fps as the GPU warms, so it sustains
  ≈400 with no margin.

## Videos

- [black_man_short_beard_lineage.mp4](black_man_short_beard_lineage.mp4)
- [black_woman_lineage.mp4](black_woman_lineage.mp4)
- [east_asian_man_goatee_lineage.mp4](east_asian_man_goatee_lineage.mp4)
- [middle_eastern_man_full_beard_lineage.mp4](middle_eastern_man_full_beard_lineage.mp4)
- [south_asian_woman_lineage.mp4](south_asian_woman_lineage.mp4)
- [white_man_clean_shaven_lineage.mp4](white_man_clean_shaven_lineage.mp4)

Each video is 2000 × ~1680 px at 24 fps, 10 s, with the identity's speech; heights differ by a few px per identity.
Each identity also has `<id>_stills.png`, taken from the raw canvas rather than the mp4, so pixel values are exact:
- the widest mouth opening;
- the frame where r5's output differs most from BEFORE.

**How to read a video:**

| Row | Content |
|---|---|
| Labels | Name, measured fps with the speedup over BEFORE's 252.0, what changed |
| Full frame | 512×896 scaled to 400×700 |
| Mouth zoom | 2.3–2.7× nearest neighbour; the factor is printed. The crop follows BEFORE's lips (centre smoothed over ±12 frames) and is identical in every column. Lips keep ≥ 6.5 px of margin. The viewport moves in 1–3 px steps, so judge flicker from the metrics or the full-frame row. |
| Output diff | The face + neck region of the composited output, i.e. what the viewer sees, including the chin warp. The BEFORE column shows the pixels. Round columns show black where identical and 40 + 8 × \|output − BEFORE\| (per channel, LSB) elsewhere: 1 LSB is drawn at 48/255, and 27 LSB or more saturates. The mapping is the same in every column and exaggerates tiny changes on purpose. The panel is max-pooled when downscaled, so every changed source pixel stays lit. The encode still dims some isolated 1-LSB speckles; the lossless stills are exact. |
| Metrics | The quality tool's numbers vs BEFORE on raw frames, with verdicts. Green = pass, amber = only the proposed landmark bar passes, orange = fail. The UNet latent gate is shown too. The BEFORE column lists the thresholds. |

**Codec noise.** In the full-frame and zoom rows it is about 2 LSB mean, 5 LSB p99 and up to ~15 LSB (crf 12), which
is larger than r2's and r5's real changes. Compare pixels with the diff row, the metrics or the stills.

## Why these are fair

- **Raw frames.** Every column is rebuilt from raw pre-encode frames with the unchanged `chin.py` compose (through
  the quality tool's `load_arm`), not from stored mp4s.
- **Bit-exact rebuilds**, checked in `lineage_report.json`:
  - BEFORE matches its render's `raw_refined_sha256`, 6/6;
  - each round matches the `raw_refined_sha256` its harness worker recorded in the round's video-capture run
    (`V_<set>.json`), 24/24. The same hashes appear in every clip of the timed T runs, so the video shows exactly
    the frames that were timed.
- **Independent checks.** Two verification passes by independent agents reproduced the hashes, the frame alignment
  (lag 0 vs ±1) and the shared inputs (`chin.py` `fd753e7d…`, the same cache and audio). Their findings are
  saved in `verification/`.
- **One encode for all.** Measured against the lossless stills, the columns are equally faithful to within ~0.2 dB
  (≤ 0.08 dB away from the column dividers).
- **The older A/B videos favour the candidate.** The per-round A/B videos (`r2_*/`, `r3_*/`, `r4_*/`) pair the stored
  pre-change mp4 (crf 18, 1.5–1.9 Mbit/s) with crf-12 captures (4.9–5.8 Mbit/s). On the r4 set, that put B
  0.56–0.78 dB closer to its raw frames over the full frame and 0.73–1.22 dB in the mouth. Use these lineage videos
  for visual judgement.

## What the numbers say (6 identities, raw frames vs BEFORE)

| Round | fps (×BEFORE) | Lip corr | Aperture delta (px) | Mouth flicker | Landmark mean / p99 (px) | Landmark repo gate / proposed bar | Other quality-tool gates | UNet latent gate (main) | Mouth PSNR mean / worst frame (dB) |
|---|---|---|---|---|---|---|---|---|---|
| r2 | 350.2 (1.39×) | 0.9998–1.0000 | 0.04–0.08 | 0.9996–1.0000 | 0.036–0.094 / 0.11–0.47 | 2/6 / 5/6 | 6/6 | pass (0.0025 / 0.39) | 55.1–57.7 / 46.9–53.9 |
| r3 | 415.6 (1.65×) | 0.9969–0.9986 | 0.23–0.37 | 1.017–1.025 | 0.20–0.36 / 0.65–1.01 | 0/6 / 0/6 | 6/6 | fail (0.038 / 2.39) | 40.2–44.2 / 36.5–40.6 |
| r4 | 414.9 (1.65×) | 0.9989–0.9994 | 0.13–0.24 | 0.998–1.003 | 0.105–0.171 / 0.37–0.65 | 0/6 / 0/6 | 6/6 | fail (0.0071 / 1.28) | 47.7–50.1 / 35.3–39.8 |
| **r5** | **≈400 (1.59×)** | 0.9997–0.9998 | 0.08–0.14 | 0.999–1.001 | 0.064–0.126 / 0.20–0.47 | 0/6 / **4/6** | 6/6 | fail on max_abs (0.0044 / 0.78) | 51.0–53.4 / 43.1–47.0 |
| ±1 LSB noise (reference) | — | 0.99985–1.0000 | 0.04–0.07 | 1.003–1.008 | 0.045–0.088 / 0.13–0.33 | — | — | — | 51.5–51.9 / 51.4–51.6 |

**Gates and bars:**
- **Landmark repo gate (0.05 / 0.15 px).** It sits at FaceMesh's own noise floor. The 0.10 / 0.35 px bar is a
  report-only proposal awaiting your decision.
- **Decoder gate.** r2–r5 share the TensorRT TAESD decoder. Its gate records FAIL on the 3 LSB max bar (max 5 LSB;
  the 0.066 mean passes). It has been used since r2, and your decision on it is still pending.
- **Mean vs worst frames.** r5's mean mouth change is at the ±1 LSB-noise level. Its worst frames are 4–8 dB below
  that, and its aperture and landmark changes are about 1.5–2× the noise arm.
- **Order of the rounds.** By mean error the order is r2 < r5 < r4 < r3. Single frames can invert it; r4 is locally
  worse than r3 on a few frames.

## Regenerate

```
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/video_lineage.py --crf 12
```

The default arms are r2–r5. `--arm 'name|fps|what|capture_dir|quality_label'` replaces them. The exact command and
the script sha256 are recorded in `lineage_report.json`.
