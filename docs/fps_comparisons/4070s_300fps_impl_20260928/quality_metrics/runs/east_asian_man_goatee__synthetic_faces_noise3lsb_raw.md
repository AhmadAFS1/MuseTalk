### east_asian_man_goatee__synthetic_faces_noise3lsb_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise3lsb` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0879 | 0.9998 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 9.0228 | 0.0777 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9452 | 9.1807 | 1.0263 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3084 | 6.3384 | 1.0048 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1938 | 3.3519 | 1.0495 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1146 | 12.6110 | 1.0410 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2627.6 | 1.0137 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 5.6 / 6.0 | 0.4780 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 3.0 | 0.5506 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 10.0 | 0.0016 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 2.6015 | -0.0009 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.1275 | -0.0073 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0866 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2452 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 59.13 / face 51.72 | mouth 45.40; worst face 50.69 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9995 / face 0.9975 | mouth 0.9924; worst mouth 0.9913 | report |  |
| Mouth sharpness (Laplacian var) | 26.3572 | 34.1105 | 1.2942 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.8729 | 0.0010 | report |  |
