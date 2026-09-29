### middle_eastern_man_full_beard__codec_raw_vs_crf18 — INCOMPLETE (profile e1, frames raw_vs_video, 240 frames)

A = `A_refined_raw` (refined), B = `A_refined_mp4` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0737 | 0.9978 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 7.9790 | 0.2794 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 10.0615 | 9.9375 | 0.9877 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.2942 | 8.7935 | 0.9461 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 5.2537 | 4.9454 | 0.9413 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.1075 | 12.9382 | 0.9871 | report |  |
| Mouth-box >6 Hz temporal power | 2938.6 | 2862.8 | 0.9742 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 30.0 / 40.0 | 2.6955 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 28.6 / 31.0 | 2.8701 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 38.0 | 2.0877 | report |  |
| Protected-lip max RGB change vs own standard | 0 | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 2.0367 | 2.0825 | 0.0458 | <= A (2.0367) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 1.8397 | 1.9876 | 0.1479 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.3534 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 1.0499 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 39.41 / face 37.58 | mouth 37.25; worst face 34.90 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9760 / face 0.9750 | mouth 0.9817; worst mouth 0.9755 | report |  |
| Mouth sharpness (Laplacian var) | 22.8754 | 31.7364 | 1.3874 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 44.4405 | 0.9413 | report |  |
