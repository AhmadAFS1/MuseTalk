### east_asian_man_goatee__refined_vs_standard_video — FAIL (profile e1, frames video, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0863 | 0.9915 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.9433 | 8.8498 | 0.5605 | <= 0.5 | FAIL |
| Flicker mouth mean abs(dt) RGB | 8.8318 | 8.8513 | 1.0022 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.9886 | 5.1753 | 0.8642 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 4.1696 | 3.8044 | 0.9124 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.0087 | 12.0084 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 2541.5 | 2404.7 | 0.9462 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 112.0 / 117.0 | 4.1947 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 101.2 / 108.0 | 5.5328 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 81.0 | 0.9847 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 2.6824 | 0.8641 | -1.8183 | <= A (2.6824) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 0.1106 | 0.8037 | 0.6931 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.9224 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 3.8079 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 38.93 / face 32.79 | mouth 39.32; worst face 27.92 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9792 / face 0.9652 | mouth 0.9800; worst mouth 0.9518 | report |  |
| Mouth sharpness (Laplacian var) | 36.7824 | 36.6311 | 0.9959 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.2110 | 54.3455 | 0.1549 | report |  |
