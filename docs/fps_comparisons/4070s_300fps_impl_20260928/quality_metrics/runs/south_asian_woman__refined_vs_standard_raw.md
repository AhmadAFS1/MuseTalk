### south_asian_woman__refined_vs_standard_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1040 | 0.9945 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.0478 | 0.4369 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2597 | 7.2399 | 0.9973 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.3991 | 3.7887 | 0.8612 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.4063 | 3.3891 | 0.9949 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7543 | 9.7294 | 0.9974 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 1882.0 | 0.9536 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 70.2 / 73.0 | 1.9185 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 68.6 / 73.0 | 2.7269 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 30.0 | 0.0058 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.3558 | 1.1245 | -1.2313 | <= A (2.3558) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.8213 | 0.5199 | -0.3015 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.7042 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 3.1543 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 47.39 / face 38.74 | mouth 40.98; worst face 32.85 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9985 / face 0.9893 | mouth 0.9891; worst mouth 0.9575 | report |  |
| Mouth sharpness (Laplacian var) | 30.6538 | 29.6339 | 0.9667 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 45.1409 | 0.1156 | report |  |
