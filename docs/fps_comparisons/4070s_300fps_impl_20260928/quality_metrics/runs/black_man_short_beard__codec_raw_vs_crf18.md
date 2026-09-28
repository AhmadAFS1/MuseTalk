### black_man_short_beard__codec_raw_vs_crf18 — INCOMPLETE (profile e1, frames raw_vs_video, 240 frames)

A = `A_refined_raw` (refined), B = `A_refined_mp4` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1289 | 0.9981 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 14.0845 | 0.3802 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.0467 | 7.9260 | 0.9850 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5824 | 5.1417 | 0.9211 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.3707 | 3.1370 | 0.9307 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8506 | 10.7275 | 0.9887 | report |  |
| Mouth-box >6 Hz temporal power | 2233.5 | 2186.7 | 0.9791 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 26.6 / 29.0 | 2.5472 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 21.2 / 23.0 | 2.6135 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 33.0 | 2.0426 | report |  |
| Protected-lip max RGB change vs own standard | 0 | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.7195 | 1.7268 | 0.0072 | <= A (1.7195) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 3.5430 | 3.5781 | 0.0351 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.3310 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.9868 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 39.73 / face 37.98 | mouth 37.47; worst face 35.38 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9758 / face 0.9738 | mouth 0.9821; worst mouth 0.9773 | report |  |
| Mouth sharpness (Laplacian var) | 32.9643 | 41.6706 | 1.2641 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 35.4243 | 0.9409 | report |  |
