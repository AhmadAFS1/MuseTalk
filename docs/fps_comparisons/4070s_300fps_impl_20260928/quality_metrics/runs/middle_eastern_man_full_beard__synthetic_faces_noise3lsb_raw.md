### middle_eastern_man_full_beard__synthetic_faces_noise3lsb_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise3lsb` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0739 | 0.9997 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 7.9992 | 0.1035 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 10.0662 | 10.2652 | 1.0198 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.3001 | 9.3410 | 1.0044 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 5.2553 | 5.3599 | 1.0199 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.1124 | 13.5359 | 1.0323 | report |  |
| Mouth-box >6 Hz temporal power | 2938.6 | 2979.4 | 1.0139 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 11.0 / 12.0 | 0.4922 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 5.0 / 7.0 | 0.5546 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 13.0 | 0.0017 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.0367 | 2.0327 | -0.0040 | <= A (2.0367) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.8397 | 1.8515 | 0.0118 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1226 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.4719 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 57.84 / face 51.13 | mouth 45.30; worst face 49.37 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9995 / face 0.9976 | mouth 0.9928; worst mouth 0.9913 | report |  |
| Mouth sharpness (Laplacian var) | 22.8633 | 29.5985 | 1.2946 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 45.1159 | 0.0007 | report |  |
