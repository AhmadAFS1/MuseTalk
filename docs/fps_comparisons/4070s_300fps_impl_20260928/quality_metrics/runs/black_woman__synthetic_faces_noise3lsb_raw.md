### black_woman__synthetic_faces_noise3lsb_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise3lsb` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0871 | 0.9997 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.7381 | 0.1039 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3175 | 7.5703 | 1.0346 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1242 | 4.1529 | 1.0070 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9751 | 4.1119 | 1.0344 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5283 | 10.0789 | 1.0578 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1743.7 | 1.0172 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 6.6 / 11.0 | 0.4549 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 5.0 / 5.0 | 0.5527 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 10.0 | 0.0017 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 1.6088 | -0.0141 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.3672 | 0.0704 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1023 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2966 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 59.18 / face 51.93 | mouth 45.77; worst face 51.09 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9995 / face 0.9976 | mouth 0.9915; worst mouth 0.9905 | report |  |
| Mouth sharpness (Laplacian var) | 21.3639 | 28.2398 | 1.3219 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2924 | 0.0009 | report |  |
