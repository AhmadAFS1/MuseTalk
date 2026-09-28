### east_asian_man_goatee__self_raw — PASS (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `A_refined_raw_again` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **True**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0879 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 9.0220 | 0.0000 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9477 | 8.9477 | 1.0000 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3086 | 6.3086 | 1.0000 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1932 | 3.1932 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1184 | 12.1184 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2592.2 | 1.0000 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 0.0 | 0.0000 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 2.6024 | 0.0000 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.1348 | 0.0000 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0000 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.0000 | <= 0.15 | PASS |
| PSNR A vs B, dB mean frame (cap 100) | - | full 100.00 / face 100.00 | mouth 100.00; worst face 100.00 | report-only | REPORT |
| SSIM A vs B mean | - | full 1.0000 / face 1.0000 | mouth 1.0000; worst mouth 1.0000 | report |  |
| Mouth sharpness (Laplacian var) | 26.3716 | 26.3716 | 1.0000 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.8736 | 0.0000 | report |  |
