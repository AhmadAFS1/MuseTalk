### south_asian_woman__refined_vs_standard_video — INCOMPLETE (profile e1, frames video, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1044 | 0.1025 | 0.9944 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.0844 | 9.9075 | 0.4313 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2319 | 7.2294 | 0.9997 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.0710 | 3.6098 | 0.8867 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2558 | 3.3159 | 1.0184 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.8016 | 9.8094 | 1.0008 | report |  |
| Mouth-box >6 Hz temporal power | 1938.8 | 1851.8 | 0.9551 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 74.2 / 81.0 | 2.9096 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 74.2 / 81.0 | 3.5166 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 30.0 | 1.1238 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 2.4546 | 1.2516 | -1.2030 | <= A (2.4546) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 0.6806 | 0.2488 | -0.4318 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.7342 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 3.1887 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 41.84 / face 37.02 | mouth 37.71; worst face 32.43 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9790 / face 0.9740 | mouth 0.9736; worst mouth 0.9458 | report |  |
| Mouth sharpness (Laplacian var) | 39.1253 | 38.5381 | 0.9850 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 44.5969 | 44.5190 | 0.1139 | report |  |
