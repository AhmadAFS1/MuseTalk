### black_woman__refined_vs_standard_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0860 | 0.9943 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.6176 | 0.4809 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3121 | 7.2647 | 0.9935 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1017 | 3.5435 | 0.8639 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.8853 | 3.7511 | 0.9654 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5190 | 9.4585 | 0.9936 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1636.6 | 0.9547 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 70.2 / 76.0 | 1.6197 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 69.6 / 76.0 | 2.6480 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 40.0 | 0.0145 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 0.6895 | -0.9333 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.6671 | 0.3703 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.6583 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 2.5708 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 46.78 / face 39.71 | mouth 42.35; worst face 35.17 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9976 / face 0.9902 | mouth 0.9916; worst mouth 0.9687 | report |  |
| Mouth sharpness (Laplacian var) | 21.3648 | 19.7610 | 0.9249 | >= 0.95 | FAIL |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2053 | 0.1394 | report |  |
