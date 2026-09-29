### east_asian_man_goatee__synthetic_faces_sparse1lsb20pct_raw — PASS (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_sparse1lsb20pct` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0879 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 9.0230 | 0.0350 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9459 | 8.9591 | 1.0015 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3074 | 6.3100 | 1.0004 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1940 | 3.2031 | 1.0029 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1159 | 12.1438 | 1.0023 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2593.7 | 1.0006 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0807 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0912 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 6.0 | 0.0007 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 2.6030 | 0.0006 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.1261 | -0.0087 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0398 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1195 | <= 0.15 | PASS |
| PSNR A vs B, dB mean frame (cap 100) | - | full 68.70 / face 61.32 | mouth 55.76; worst face 58.62 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9999 / face 0.9996 | mouth 0.9988; worst mouth 0.9985 | report |  |
| Mouth sharpness (Laplacian var) | 26.3611 | 26.8087 | 1.0170 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.8734 | 0.0003 | report |  |
