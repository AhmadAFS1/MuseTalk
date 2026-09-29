### south_asian_woman__srcblkA8 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcblkA8` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1054 | 0.9994 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.1793 | 0.1481 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2720 | 7.2590 | 0.9982 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.4880 | 4.4845 | 0.9992 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2683 | 3.2659 | 0.9993 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7736 | 9.7550 | 0.9981 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 1967.7 | 0.9970 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 5.0 / 6.0 | 0.0896 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 6.0 | 0.1072 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 5.0 | 0.0006 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.3558 | 2.3407 | -0.0151 | <= A (2.3558) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.8213 | 0.7682 | -0.0531 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1049 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.3737 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.1049 / 0.3737 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 66.10 / face 57.40 | mouth 49.63; worst face 46.72 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9995 | mouth 0.9977; worst mouth 0.9863 | report |  |
| Mouth sharpness (Laplacian var) | 30.7483 | 30.9025 | 1.0050 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 45.2189 | 0.0023 | report |  |
