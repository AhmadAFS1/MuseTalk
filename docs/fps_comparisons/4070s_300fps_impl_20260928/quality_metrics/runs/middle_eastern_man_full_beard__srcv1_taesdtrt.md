### middle_eastern_man_full_beard__srcv1_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcv1_taesdtrt` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0737 | 0.9973 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 7.9783 | 0.3245 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 10.0626 | 10.2705 | 1.0207 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.2972 | 9.3620 | 1.0070 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 5.2598 | 5.3430 | 1.0158 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.1086 | 13.5578 | 1.0343 | report |  |
| Mouth-box >6 Hz temporal power | 2938.6 | 3038.9 | 1.0341 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 35.3 / 42.0 | 0.6687 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 26.6 / 42.0 | 0.8470 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 52.0 | 0.0064 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.0367 | 2.0524 | 0.0157 | <= A (2.0367) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.8397 | 1.8064 | -0.0333 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.3551 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.9810 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.3551 / 0.9810 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 53.56 / face 46.85 | mouth 40.19; worst face 41.00 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9993 / face 0.9967 | mouth 0.9890; worst mouth 0.9763 | report |  |
| Mouth sharpness (Laplacian var) | 22.9094 | 22.9754 | 1.0029 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 45.1235 | 0.0290 | report |  |
