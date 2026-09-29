### south_asian_woman__srcmix_taesdtrt — PASS (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcmix_taesdtrt` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1053 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.1756 | 0.0399 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2729 | 7.2755 | 1.0004 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.4899 | 4.4893 | 0.9999 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2685 | 3.2685 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7745 | 9.7795 | 1.0005 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 1974.5 | 1.0005 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0361 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0440 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 2.0 | 0.0002 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.3558 | 2.3554 | -0.0004 | <= A (2.3558) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.8213 | 0.8194 | -0.0019 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0360 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1101 | <= 0.15 | PASS |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0360 / 0.1101 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 72.85 / face 64.15 | mouth 57.69; worst face 61.31 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9998 | mouth 0.9993; worst mouth 0.9988 | report |  |
| Mouth sharpness (Laplacian var) | 30.7554 | 30.7685 | 1.0004 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 45.2196 | 0.0036 | report |  |
