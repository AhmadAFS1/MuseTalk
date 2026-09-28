### east_asian_man_goatee__srcfp16_taesdtrt — PASS (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcfp16_taesdtrt` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0879 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 9.0171 | 0.0392 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9458 | 8.9441 | 0.9998 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3086 | 6.3051 | 0.9994 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1934 | 3.1940 | 1.0002 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1158 | 12.1153 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2592.3 | 1.0001 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0488 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0574 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 4.0 | 0.0007 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 2.5973 | -0.0051 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.1503 | 0.0155 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0422 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1201 | <= 0.15 | PASS |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0422 / 0.1201 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 70.22 / face 62.85 | mouth 56.44; worst face 58.11 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9997 | mouth 0.9992; worst mouth 0.9984 | report |  |
| Mouth sharpness (Laplacian var) | 26.3589 | 26.3856 | 1.0010 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.8726 | 0.0056 | report |  |
