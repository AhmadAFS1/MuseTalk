### black_woman__srcblkA8 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcblkA8` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0879 | 0.9989 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.8245 | 0.2003 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3168 | 7.3102 | 0.9991 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1258 | 4.1277 | 1.0005 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9755 | 3.9741 | 0.9996 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5274 | 9.5152 | 0.9987 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1708.9 | 0.9969 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 7.2 / 11.0 | 0.1118 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 5.0 | 0.1347 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 16.0 | 0.0020 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 1.6185 | -0.0043 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.3689 | 0.0721 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1405 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.5550 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.1405 / 0.5550 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 63.68 / face 56.47 | mouth 49.11; worst face 45.27 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9994 | mouth 0.9974; worst mouth 0.9863 | report |  |
| Mouth sharpness (Laplacian var) | 21.3626 | 21.4704 | 1.0050 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2916 | 0.0041 | report |  |
