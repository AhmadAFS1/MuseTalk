### black_woman__codec_raw_vs_crf18 — INCOMPLETE (profile e1, frames raw_vs_video, 240 frames)

A = `A_refined_raw` (refined), B = `A_refined_mp4` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0871 | 0.9981 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.7355 | 0.2841 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3164 | 7.2729 | 0.9941 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1228 | 3.8644 | 0.9373 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9746 | 3.8953 | 0.9800 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5277 | 9.5330 | 1.0006 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1682.6 | 0.9815 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 24.0 / 29.0 | 2.3593 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 20.6 / 24.0 | 2.4292 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 33.0 | 2.1949 | report |  |
| Protected-lip max RGB change vs own standard | 0 | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.6228 | 1.6703 | 0.0475 | <= A (1.6228) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 0.2968 | 0.1888 | -0.1080 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.2879 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.7903 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 39.34 / face 38.19 | mouth 37.87; worst face 35.70 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9756 / face 0.9760 | mouth 0.9831; worst mouth 0.9772 | report |  |
| Mouth sharpness (Laplacian var) | 21.3620 | 29.2953 | 1.3714 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 41.6054 | 0.8854 | report |  |
