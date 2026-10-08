### east_asian_man_goatee__portable_ref2 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `portable_ref2` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0878 | 0.9998 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 9.0122 | 0.0884 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9449 | 8.9455 | 1.0001 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3081 | 6.3098 | 1.0003 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1937 | 3.1948 | 1.0003 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1144 | 12.1153 | 1.0001 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2592.3 | 1.0001 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 7.0 / 8.0 | 0.0840 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 6.6 / 8.0 | 0.0973 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 9.0 | 0.0013 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 2.5998 | -0.0026 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.1369 | 0.0021 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0735 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2218 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0735 / 0.2218 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 66.89 / face 59.51 | mouth 52.19; worst face 55.85 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9996 | mouth 0.9985; worst mouth 0.9973 | report |  |
| Mouth sharpness (Laplacian var) | 26.3544 | 26.2733 | 0.9969 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.8727 | 0.0063 | report |  |
