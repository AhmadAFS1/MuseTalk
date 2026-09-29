### black_man_short_beard__srcv1_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcv1_taesdtrt` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1284 | 0.9979 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 14.0321 | 0.3679 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.0475 | 8.2167 | 1.0210 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5839 | 5.5785 | 0.9990 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.3783 | 3.4216 | 1.0128 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8486 | 11.2196 | 1.0342 | report |  |
| Mouth-box >6 Hz temporal power | 2233.5 | 2301.2 | 1.0303 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 18.6 / 20.0 | 0.4931 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 16.6 / 19.0 | 0.5781 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 31.0 | 0.0068 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.7195 | 1.7542 | 0.0346 | <= A (1.7195) + 0.05 | PASS |
| Chin positive excess p95 (px) | 3.5430 | 3.5893 | 0.0464 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.3152 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 1.0121 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.3152 / 1.0121 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 55.56 / face 48.65 | mouth 41.28; worst face 43.66 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9995 / face 0.9977 | mouth 0.9918; worst mouth 0.9864 | report |  |
| Mouth sharpness (Laplacian var) | 33.0121 | 32.8653 | 0.9956 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 36.1212 | 0.0343 | report |  |
