### middle_eastern_man_full_beard__srcmix_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcmix_taesdtrt` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0740 | 0.9998 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 8.0089 | 0.0756 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 10.0668 | 10.0633 | 0.9996 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.3011 | 9.3049 | 1.0004 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 5.2546 | 5.2544 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.1130 | 13.1080 | 0.9996 | report |  |
| Mouth-box >6 Hz temporal power | 2938.6 | 2937.8 | 0.9997 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 7.0 / 11.0 | 0.0770 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 5.0 | 0.0957 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 13.0 | 0.0011 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.0367 | 2.0682 | 0.0315 | <= A (2.0367) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.8397 | 1.8272 | -0.0125 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0936 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.4749 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0936 / 0.4749 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 67.45 / face 60.75 | mouth 55.09; worst face 53.14 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9997 | mouth 0.9990; worst mouth 0.9972 | report |  |
| Mouth sharpness (Laplacian var) | 22.8325 | 22.8428 | 1.0004 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 45.1159 | 0.0026 | report |  |
