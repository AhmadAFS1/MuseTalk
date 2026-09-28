### east_asian_man_goatee__srcmix_taesdtrt — PASS (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcmix_taesdtrt` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0879 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 9.0203 | 0.0427 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9463 | 8.9467 | 1.0000 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3087 | 6.3075 | 0.9998 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1936 | 3.1948 | 1.0004 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1167 | 12.1197 | 1.0002 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2592.9 | 1.0003 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 5.0 | 0.0493 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.6 / 5.0 | 0.0577 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 5.0 | 0.0007 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 2.5972 | -0.0052 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.1261 | -0.0087 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0412 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1230 | <= 0.15 | PASS |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0412 / 0.1230 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 70.17 / face 62.79 | mouth 56.39; worst face 56.43 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9997 | mouth 0.9992; worst mouth 0.9981 | report |  |
| Mouth sharpness (Laplacian var) | 26.3603 | 26.3811 | 1.0008 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.8726 | 0.0055 | report |  |
