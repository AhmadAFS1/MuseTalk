### black_man_short_beard__srcblkA8 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcblkA8` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1301 | 0.9990 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 14.2104 | 0.2396 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.0504 | 8.0546 | 1.0005 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5865 | 5.5765 | 0.9982 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.3731 | 3.3737 | 1.0002 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8532 | 10.9042 | 1.0047 | report |  |
| Mouth-box >6 Hz temporal power | 2233.5 | 2246.0 | 1.0056 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 9.0 / 14.0 | 0.1469 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 6.6 / 14.0 | 0.1628 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 13.0 | 0.0030 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.7195 | 1.7110 | -0.0085 | <= A (1.7195) + 0.05 | PASS |
| Chin positive excess p95 (px) | 3.5430 | 3.6101 | 0.0672 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1610 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.6465 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.1610 / 0.6465 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 62.12 / face 55.23 | mouth 47.65; worst face 47.10 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9998 / face 0.9993 | mouth 0.9973; worst mouth 0.9909 | report |  |
| Mouth sharpness (Laplacian var) | 32.9944 | 33.0881 | 1.0028 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 36.1272 | 0.0028 | report |  |
