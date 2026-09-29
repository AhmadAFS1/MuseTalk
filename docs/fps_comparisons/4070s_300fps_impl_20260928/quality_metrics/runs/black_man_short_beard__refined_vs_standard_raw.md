### black_man_short_beard__refined_vs_standard_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1277 | 0.9917 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 13.9562 | 0.7163 | <= 0.5 | FAIL |
| Flicker mouth mean abs(dt) RGB | 8.0382 | 8.0347 | 0.9996 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5830 | 4.3653 | 0.7819 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 4.0211 | 3.6868 | 0.9169 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8357 | 10.8158 | 0.9982 | report |  |
| Mouth-box >6 Hz temporal power | 2223.2 | 2110.7 | 0.9494 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 93.6 / 95.0 | 2.2395 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 73.6 / 82.0 | 2.9790 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 47.0 | 0.0208 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.7195 | 0.4743 | -1.2452 | <= A (1.7195) + 0.05 | PASS |
| Chin positive excess p95 (px) | 3.5430 | 0.9789 | -2.5641 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.7490 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 3.1547 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 43.60 / face 36.86 | mouth 46.95; worst face 31.31 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9935 / face 0.9711 | mouth 0.9933; worst mouth 0.9685 | report |  |
| Mouth sharpness (Laplacian var) | 33.0363 | 28.7051 | 0.8689 | >= 0.95 | FAIL |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 36.1359 | 0.0601 | report |  |
