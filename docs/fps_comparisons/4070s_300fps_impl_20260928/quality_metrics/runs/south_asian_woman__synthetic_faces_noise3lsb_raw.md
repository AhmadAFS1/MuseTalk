### south_asian_woman__synthetic_faces_noise3lsb_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise3lsb` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1054 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.1810 | 0.0679 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2730 | 7.5447 | 1.0374 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.4874 | 4.5046 | 1.0038 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2686 | 3.4258 | 1.0481 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7744 | 10.3495 | 1.0588 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 2001.9 | 1.0144 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 4.0 | 0.4534 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 4.0 | 0.5562 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 4.0 | 0.0005 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.3558 | 2.3518 | -0.0040 | <= A (2.3558) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.8213 | 0.7661 | -0.0553 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0716 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2123 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 60.78 / face 52.06 | mouth 45.72; worst face 51.26 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9997 / face 0.9976 | mouth 0.9916; worst mouth 0.9891 | report |  |
| Mouth sharpness (Laplacian var) | 30.7544 | 39.9443 | 1.2988 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 45.2184 | 0.0009 | report |  |
