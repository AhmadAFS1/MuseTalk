### black_man_short_beard__synthetic_faces_noise3lsb_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise3lsb` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1293 | 0.9998 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 14.1321 | 0.1222 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.0499 | 8.2954 | 1.0305 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5872 | 5.6290 | 1.0075 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.3717 | 3.5132 | 1.0420 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8557 | 11.3713 | 1.0475 | report |  |
| Mouth-box >6 Hz temporal power | 2233.5 | 2272.3 | 1.0174 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 7.0 / 9.0 | 0.4846 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.5603 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 11.0 | 0.0024 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.7195 | 1.7167 | -0.0028 | <= A (1.7195) + 0.05 | PASS |
| Chin positive excess p95 (px) | 3.5430 | 3.5856 | 0.0426 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1129 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.3401 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 58.32 / face 51.40 | mouth 45.38; worst face 50.12 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9995 / face 0.9975 | mouth 0.9922; worst mouth 0.9902 | report |  |
| Mouth sharpness (Laplacian var) | 32.9750 | 39.9475 | 1.2115 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 36.1264 | 0.0010 | report |  |
