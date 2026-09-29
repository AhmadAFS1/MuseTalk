### south_asian_woman__synthetic_faces_noise1lsb_raw — PASS (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise1lsb` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1053 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.1762 | 0.0442 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2730 | 7.3237 | 1.0070 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.4889 | 4.4915 | 1.0006 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2688 | 3.2983 | 1.0090 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7743 | 9.8855 | 1.0114 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 1978.5 | 1.0025 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 3.0 | 0.1888 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.2313 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 3.0 | 0.0002 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.3558 | 2.3439 | -0.0119 | <= A (2.3558) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.8213 | 0.7533 | -0.0680 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0446 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1284 | <= 0.15 | PASS |
| PSNR A vs B, dB mean frame (cap 100) | - | full 66.53 / face 57.82 | mouth 51.88; worst face 57.06 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9999 / face 0.9991 | mouth 0.9970; worst mouth 0.9962 | report |  |
| Mouth sharpness (Laplacian var) | 30.7535 | 32.4459 | 1.0550 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 45.2185 | 0.0006 | report |  |
