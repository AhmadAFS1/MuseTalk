### south_asian_woman__portable_ref2 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `portable_ref2` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1053 | 0.9998 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.1716 | 0.0863 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2725 | 7.2868 | 1.0020 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.4888 | 4.4885 | 0.9999 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2685 | 3.2689 | 1.0001 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7735 | 9.8060 | 1.0033 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 1983.0 | 1.0048 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.6 / 4.0 | 0.0623 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0758 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 4.0 | 0.0004 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.3558 | 2.3426 | -0.0132 | <= A (2.3558) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.8213 | 0.7895 | -0.0318 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0648 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2086 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0648 / 0.2086 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 69.17 / face 60.47 | mouth 53.24; worst face 55.65 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9997 | mouth 0.9986; worst mouth 0.9974 | report |  |
| Mouth sharpness (Laplacian var) | 30.7511 | 30.8328 | 1.0027 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 45.2199 | 0.0032 | report |  |
