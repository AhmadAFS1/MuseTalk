### black_man_short_beard__portable_ref2 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `portable_ref2` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1297 | 0.9997 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 14.1662 | 0.1298 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.0514 | 8.0547 | 1.0004 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5865 | 5.5837 | 0.9995 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.3724 | 3.3706 | 0.9995 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8552 | 10.8575 | 1.0002 | report |  |
| Mouth-box >6 Hz temporal power | 2233.5 | 2232.8 | 0.9997 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 5.6 / 7.0 | 0.0987 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 4.0 | 0.1112 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 11.0 | 0.0020 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.7195 | 1.7172 | -0.0023 | <= A (1.7195) + 0.05 | PASS |
| Chin positive excess p95 (px) | 3.5430 | 3.5665 | 0.0235 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1054 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.3414 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.1054 / 0.3414 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 65.20 / face 58.32 | mouth 51.37; worst face 54.20 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9995 | mouth 0.9983; worst mouth 0.9968 | report |  |
| Mouth sharpness (Laplacian var) | 32.9812 | 32.9534 | 0.9992 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 36.1268 | 0.0036 | report |  |
