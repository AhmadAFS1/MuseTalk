### south_asian_woman__srcfp16_taesdtrt — PASS (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcfp16_taesdtrt` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1053 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.1734 | 0.0451 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2734 | 7.2760 | 1.0004 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.4897 | 4.4890 | 0.9999 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2686 | 3.2689 | 1.0001 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7754 | 9.7800 | 1.0005 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 1975.0 | 1.0007 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0361 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 2.0 | 0.0437 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 3.0 | 0.0002 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.3558 | 2.3535 | -0.0023 | <= A (2.3558) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.8213 | 0.8042 | -0.0171 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0359 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1096 | <= 0.15 | PASS |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0359 / 0.1096 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 72.86 / face 64.17 | mouth 57.73; worst face 61.41 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9998 | mouth 0.9993; worst mouth 0.9988 | report |  |
| Mouth sharpness (Laplacian var) | 30.7587 | 30.7757 | 1.0006 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 45.2196 | 0.0037 | report |  |
