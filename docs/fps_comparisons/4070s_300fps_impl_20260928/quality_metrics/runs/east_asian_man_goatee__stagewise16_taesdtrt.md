### east_asian_man_goatee__stagewise16_taesdtrt — PASS (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `stagewise16_taesdtrt` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0880 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 9.0241 | 0.0432 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9462 | 8.9451 | 0.9999 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3091 | 6.3046 | 0.9993 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1936 | 3.1936 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1161 | 12.1160 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2593.8 | 1.0006 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 5.0 | 0.0497 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 3.0 | 0.0580 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 5.0 | 0.0008 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 2.6057 | 0.0033 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.1149 | -0.0199 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0430 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1280 | <= 0.15 | PASS |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0430 / 0.1280 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 70.12 / face 62.75 | mouth 56.44; worst face 57.52 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9997 | mouth 0.9992; worst mouth 0.9980 | report |  |
| Mouth sharpness (Laplacian var) | 26.3577 | 26.3767 | 1.0007 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.8727 | 0.0054 | report |  |
