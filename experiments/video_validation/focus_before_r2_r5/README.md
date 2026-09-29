# Focus videos: BEFORE (previous working pipeline) vs r2 (previous round, 350 fps) vs r5 (new, ≈400 fps)

These are made the same way as [`../lineage_all_rounds/`](../lineage_all_rounds/README.md), with three wider columns.
That README has the row-by-row guide, the fairness checks and the full metrics table.
- **Full frame:** native 512×896.
- **Mouth zoom:** 2.9–3.5× nearest neighbour, following the lips.
- **Face + neck output diff:** nearest-upscaled 1.07–1.31×, so every source pixel is drawn.
- **Encoding:** one encode at crf 10.

Every column is rebuilt bit-exactly from its raw frames (`lineage_report.json`: BEFORE vs `render.json`, and each round
vs its harness render SHA, all true).

| Column | What it is | Aggregate fps, 6 streams, same harness |
|---|---|---|
| BEFORE (previous working pipeline) | Shipping TensorRT FP16 bs8 UNet + compiled TAESD + 100% chin + refined seam. Pre-change renders from the accepted recipe; visual acceptance pending. | 252.0 (single stream: 148–171) |
| r2 (previous round) | Stagewise FP16 UNet + source-prefix cache + INT8 `down3`/`mid` + TensorRT TAESD | 350.2 (1.39×) |
| r5 NEW | r2 + INT8 on UNet layers chosen by error per MAC (`gmac_0.50`) | ≈400 sustained (404.0 → 399.96 over 5 × 64 s; 1.59×) |

**r5 vs BEFORE** (6 identities, raw frames):
- lip-sync correlation 0.9997–0.9998;
- aperture delta 0.08–0.14 px;
- flicker 0.999–1.001;
- landmarks 0.064–0.126 / 0.20–0.47 px: repo gate 0/6, proposed bar 4/6 (r2: 2/6 and 5/6);
- mean mouth PSNR 51.0–53.4 dB, the ±1 LSB-noise level; worst frames 43–47 dB;
- UNet latent gate: fails max_abs.

**What you'll see.** At normal viewing, BEFORE, r2 and r5 look the same. Both verification passes found no visible
difference in the full frame or the zoom.
- The diff row shows r2 as sparse, faint speckle.
- r5's changes are denser and follow the lip contours, the teeth and the jaw/beard edges.

**Codec noise.** In the full-frame and zoom rows it is larger than the real r2/r5 changes. Use the diff row, the
metrics or the lossless `<id>_stills.png` for pixel comparisons.

Videos:
- [black_man_short_beard](black_man_short_beard_lineage.mp4)
- [black_woman](black_woman_lineage.mp4)
- [east_asian_man_goatee](east_asian_man_goatee_lineage.mp4)
- [middle_eastern_man_full_beard](middle_eastern_man_full_beard_lineage.mp4)
- [south_asian_woman](south_asian_woman_lineage.mp4)
- [white_man_clean_shaven](white_man_clean_shaven_lineage.mp4)

Regenerate with:

```
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/video_lineage.py --col-width 512 --crf 10 --out-name focus_before_r2_r5 \
  --arm 'r2 (previous round)|350.2 fps aggregate (6 streams)|stagewise FP16 UNet + source-prefix cache + INT8 down3/mid + TRT TAESD|docs/fps_comparisons/4070s_300fps_impl_20260928/chin_multistream/V_srcmix_taesdtrt|srcmix_taesdtrt' \
  --arm 'r5 NEW|400 fps sustained aggregate (6 streams; 404.0 to 399.96 over 5 x 64 s)|r2 + INT8 on 117 UNet layers chosen by error per MAC (gmac_0.50)|docs/fps_comparisons/4070s_400fps_20260928/chin_multistream/V_srcg50|srcg50'
```
