### black_man_short_beard__refined_vs_standard_video — FAIL (profile e1, frames video, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1289 | 0.1279 | 0.9897 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.0845 | 13.9732 | 0.8044 | <= 0.5 | FAIL |
| Flicker mouth mean abs(dt) RGB | 7.9177 | 7.9233 | 1.0007 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.1427 | 4.1026 | 0.7978 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.7530 | 3.5049 | 0.9339 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.7117 | 10.6980 | 0.9987 | report |  |
| Mouth-box >6 Hz temporal power | 2175.0 | 2073.5 | 0.9533 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 92.0 / 93.0 | 3.2561 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 83.6 / 89.0 | 3.8537 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 44.0 | 1.0024 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.7268 | 0.4737 | -1.2531 | <= A (1.7268) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 3.5781 | 0.8113 | -2.7668 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.8063 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 3.2190 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 40.77 / face 35.76 | mouth 39.34; worst face 30.89 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9758 / face 0.9555 | mouth 0.9767; worst mouth 0.9519 | report |  |
| Mouth sharpness (Laplacian var) | 41.6923 | 38.3994 | 0.9210 | >= 0.95 | FAIL |
| Face Lab L* mean; delta = dE76 of means | 35.4243 | 35.4269 | 0.0571 | report |  |
