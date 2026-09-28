### east_asian_man_goatee__srcv1_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcv1_taesdtrt` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0877 | 0.9986 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 8.9938 | 0.2344 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9413 | 9.1474 | 1.0231 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3060 | 6.3417 | 1.0057 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1985 | 3.2487 | 1.0157 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1100 | 12.5116 | 1.0332 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2678.2 | 1.0332 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 22.2 / 27.0 | 0.3890 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 22.2 / 27.0 | 0.4798 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 27.0 | 0.0049 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 2.6077 | 0.0053 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.1165 | -0.0183 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.2215 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.6491 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.2215 / 0.6491 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 57.78 / face 50.38 | mouth 42.56; worst face 46.65 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9997 / face 0.9984 | mouth 0.9937; worst mouth 0.9795 | report |  |
| Mouth sharpness (Laplacian var) | 26.3338 | 26.5804 | 1.0094 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.8822 | 0.0336 | report |  |
