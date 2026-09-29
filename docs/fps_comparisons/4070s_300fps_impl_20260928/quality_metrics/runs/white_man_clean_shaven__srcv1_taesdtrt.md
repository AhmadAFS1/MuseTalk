### white_man_clean_shaven__srcv1_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcv1_taesdtrt` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0893 | 0.9969 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.7904 | 0.3115 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5057 | 6.6700 | 1.0253 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.2055 | 4.2416 | 1.0086 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3004 | 2.3445 | 1.0192 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5614 | 8.8732 | 1.0364 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1371.2 | 1.0464 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 13.6 / 16.0 | 0.3551 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 12.0 / 16.0 | 0.4352 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 22.0 | 0.0043 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.1328 | 1.1311 | -0.0017 | <= A (1.1328) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.6628 | 1.7332 | 0.0705 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.2498 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.8110 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.2498 / 0.8110 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 58.74 / face 51.07 | mouth 43.45; worst face 47.38 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9997 / face 0.9983 | mouth 0.9940; worst mouth 0.9868 | report |  |
| Mouth sharpness (Laplacian var) | 22.9637 | 23.1378 | 1.0076 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 53.1324 | 0.0358 | report |  |
