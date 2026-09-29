### black_woman__srcfp16_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcfp16_taesdtrt` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0872 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.7432 | 0.0565 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3188 | 7.3181 | 0.9999 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1256 | 4.1266 | 1.0002 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9745 | 3.9742 | 0.9999 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5300 | 9.5287 | 0.9999 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1713.4 | 0.9995 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0408 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0504 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 5.0 | 0.0006 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 1.6195 | -0.0033 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.2891 | -0.0077 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0510 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1542 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0510 / 0.1542 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 70.46 / face 63.28 | mouth 56.99; worst face 56.77 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9998 | mouth 0.9991; worst mouth 0.9962 | report |  |
| Mouth sharpness (Laplacian var) | 21.3658 | 21.3722 | 1.0003 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2932 | 0.0049 | report |  |
