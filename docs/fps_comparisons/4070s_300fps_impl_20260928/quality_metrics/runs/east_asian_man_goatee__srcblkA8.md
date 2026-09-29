### east_asian_man_goatee__srcblkA8 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcblkA8` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0884 | 0.9990 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 9.0670 | 0.1648 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9436 | 8.9321 | 0.9987 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3067 | 6.3174 | 1.0017 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1949 | 3.1967 | 1.0006 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1117 | 12.1120 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2593.2 | 1.0004 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 11.0 / 16.0 | 0.1231 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 10.6 / 16.0 | 0.1398 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 16.0 | 0.0025 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 2.6157 | 0.0133 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.1467 | 0.0119 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1171 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.4946 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.1171 / 0.4946 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 63.94 / face 56.56 | mouth 48.76; worst face 45.87 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9994 | mouth 0.9976; worst mouth 0.9854 | report |  |
| Mouth sharpness (Laplacian var) | 26.3394 | 26.3351 | 0.9998 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.8714 | 0.0068 | report |  |
