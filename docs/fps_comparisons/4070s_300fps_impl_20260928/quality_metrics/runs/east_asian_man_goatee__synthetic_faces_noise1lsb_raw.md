### east_asian_man_goatee__synthetic_faces_noise1lsb_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise1lsb` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0879 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 9.0207 | 0.0445 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9465 | 8.9879 | 1.0046 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3082 | 6.3107 | 1.0004 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1940 | 3.2223 | 1.0089 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1167 | 12.2053 | 1.0073 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2597.6 | 1.0021 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.2007 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 2.0 | 0.2302 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 6.0 | 0.0009 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 2.6044 | 0.0020 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.1208 | -0.0140 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0518 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1531 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 64.80 / face 57.40 | mouth 51.61; worst face 55.82 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9998 / face 0.9990 | mouth 0.9973; worst mouth 0.9967 | report |  |
| Mouth sharpness (Laplacian var) | 26.3597 | 27.7667 | 1.0534 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.8730 | 0.0006 | report |  |
