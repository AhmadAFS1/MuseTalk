### black_man_short_beard__self_raw — PASS (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `A_refined_raw_again` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **True**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1294 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 14.1362 | 0.0000 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.0532 | 8.0532 | 1.0000 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5877 | 5.5877 | 1.0000 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.3707 | 3.3707 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8607 | 10.8607 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 2233.5 | 2233.5 | 1.0000 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 0.0 | 0.0000 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.7195 | 1.7195 | 0.0000 | <= A (1.7195) + 0.05 | PASS |
| Chin positive excess p95 (px) | 3.5430 | 3.5430 | 0.0000 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0000 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.0000 | <= 0.15 | PASS |
| PSNR A vs B, dB mean frame (cap 100) | - | full 100.00 / face 100.00 | mouth 100.00; worst face 100.00 | report-only | REPORT |
| SSIM A vs B mean | - | full 1.0000 / face 1.0000 | mouth 1.0000; worst mouth 1.0000 | report |  |
| Mouth sharpness (Laplacian var) | 32.9906 | 32.9906 | 1.0000 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 36.1263 | 0.0000 | report |  |
