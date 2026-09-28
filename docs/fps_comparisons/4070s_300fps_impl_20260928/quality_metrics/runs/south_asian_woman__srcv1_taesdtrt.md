### south_asian_woman__srcv1_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcv1_taesdtrt` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1048 | 0.9982 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.1228 | 0.2759 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2686 | 7.4244 | 1.0214 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.4860 | 4.4930 | 1.0015 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2678 | 3.3033 | 1.0108 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7677 | 10.0703 | 1.0310 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 2036.2 | 1.0318 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 11.8 / 16.0 | 0.2963 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 9.0 / 16.0 | 0.3764 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 14.0 | 0.0013 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.3558 | 2.3715 | 0.0157 | <= A (2.3558) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.8213 | 0.7860 | -0.0353 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1971 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.6522 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.1971 / 0.6522 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 60.49 / face 51.78 | mouth 44.18; worst face 48.54 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9998 / face 0.9987 | mouth 0.9946; worst mouth 0.9894 | report |  |
| Mouth sharpness (Laplacian var) | 30.7140 | 30.9538 | 1.0078 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 45.2287 | 0.0357 | report |  |
