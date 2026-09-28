### east_asian_man_goatee__refined_vs_standard_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0868 | 0.9913 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 8.9106 | 0.5613 | <= 0.5 | FAIL |
| Flicker mouth mean abs(dt) RGB | 8.9156 | 8.9207 | 1.0006 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3141 | 5.3963 | 0.8546 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 4.3239 | 3.8986 | 0.9016 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.0670 | 12.0476 | 0.9984 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2439.2 | 0.9410 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 113.3 / 117.0 | 3.2384 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 100.6 / 103.0 | 4.7032 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 90.0 | 0.0178 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 0.7594 | -1.8430 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.7514 | 0.6165 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.9245 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 3.8560 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 40.66 / face 33.35 | mouth 48.30; worst face 28.10 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9961 / face 0.9805 | mouth 0.9957; worst mouth 0.9622 | report |  |
| Mouth sharpness (Laplacian var) | 26.2116 | 25.8515 | 0.9863 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 55.0090 | 0.1514 | report |  |
