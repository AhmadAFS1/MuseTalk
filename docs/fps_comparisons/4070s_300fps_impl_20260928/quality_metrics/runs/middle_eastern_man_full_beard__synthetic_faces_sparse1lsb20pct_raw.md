### middle_eastern_man_full_beard__synthetic_faces_sparse1lsb20pct_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_sparse1lsb20pct` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0740 | 0.9998 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 8.0035 | 0.0726 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 10.0668 | 10.0777 | 1.0011 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.3007 | 9.3045 | 1.0004 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 5.2544 | 5.2603 | 1.0011 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.1143 | 13.1377 | 1.0018 | report |  |
| Mouth-box >6 Hz temporal power | 2938.6 | 2940.9 | 1.0008 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 5.6 / 7.0 | 0.0825 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0926 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 10.0 | 0.0007 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.0367 | 2.0486 | 0.0119 | <= A (2.0367) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.8397 | 1.8352 | -0.0044 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0835 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.4307 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 67.56 / face 60.86 | mouth 55.70; worst face 56.00 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9999 / face 0.9996 | mouth 0.9988; worst mouth 0.9986 | report |  |
| Mouth sharpness (Laplacian var) | 22.8433 | 23.2184 | 1.0164 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 45.1160 | 0.0001 | report |  |
