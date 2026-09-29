### middle_eastern_man_full_beard__refined_vs_standard_video — FAIL (profile e1, frames video, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0737 | 0.0716 | 0.9902 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 7.9790 | 7.7486 | 0.6161 | <= 0.5 | FAIL |
| Flicker mouth mean abs(dt) RGB | 9.9310 | 9.9395 | 1.0009 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 8.7541 | 6.1458 | 0.7021 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 6.4965 | 5.9121 | 0.9100 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.9306 | 12.9314 | 1.0001 | report |  |
| Mouth-box >6 Hz temporal power | 2873.3 | 2647.8 | 0.9215 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 120.6 / 127.0 | 4.1569 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 84.8 / 110.0 | 5.2262 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 68.0 | 1.0754 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 2.0825 | 0.5438 | -1.5387 | <= A (2.0825) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 1.9876 | 1.0111 | -0.9765 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 1.1010 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 5.0079 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 38.99 / face 33.50 | mouth 39.19; worst face 29.99 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9738 / face 0.9453 | mouth 0.9787; worst mouth 0.9552 | report |  |
| Mouth sharpness (Laplacian var) | 31.8469 | 30.0039 | 0.9421 | >= 0.95 | FAIL |
| Face Lab L* mean; delta = dE76 of means | 44.4405 | 44.4773 | 0.0629 | report |  |
