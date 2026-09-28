### black_man_short_beard__synthetic_faces_noise1lsb_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise1lsb` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1295 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 14.1441 | 0.0728 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.0521 | 8.0967 | 1.0055 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5867 | 5.5936 | 1.0012 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.3715 | 3.3963 | 1.0074 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8583 | 10.9529 | 1.0087 | report |  |
| Mouth-box >6 Hz temporal power | 2233.5 | 2239.5 | 1.0027 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 5.0 / 7.0 | 0.2045 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 4.0 | 0.2358 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 6.0 | 0.0013 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.7195 | 1.7246 | 0.0051 | <= A (1.7195) + 0.05 | PASS |
| Chin positive excess p95 (px) | 3.5430 | 3.5510 | 0.0080 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0689 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2107 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 63.97 / face 57.06 | mouth 51.59; worst face 55.99 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9998 / face 0.9990 | mouth 0.9972; worst mouth 0.9965 | report |  |
| Mouth sharpness (Laplacian var) | 32.9778 | 34.2552 | 1.0387 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 36.1260 | 0.0005 | report |  |
