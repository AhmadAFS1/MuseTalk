### black_man_short_beard__synthetic_faces_sparse1lsb20pct_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_sparse1lsb20pct` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1295 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 14.1465 | 0.0617 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.0511 | 8.0649 | 1.0017 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5856 | 5.5860 | 1.0001 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.3710 | 3.3792 | 1.0024 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8575 | 10.8866 | 1.0027 | report |  |
| Mouth-box >6 Hz temporal power | 2233.5 | 2235.8 | 1.0010 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0829 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0935 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 4.0 | 0.0009 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.7195 | 1.7220 | 0.0025 | <= A (1.7195) + 0.05 | PASS |
| Chin positive excess p95 (px) | 3.5430 | 3.5220 | -0.0210 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0564 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1695 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 67.87 / face 60.98 | mouth 55.74; worst face 58.44 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9999 / face 0.9996 | mouth 0.9987; worst mouth 0.9984 | report |  |
| Mouth sharpness (Laplacian var) | 32.9795 | 33.3704 | 1.0119 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 36.1263 | 0.0002 | report |  |
