### black_man_short_beard__srcmix_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcmix_taesdtrt` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1294 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 14.1411 | 0.0647 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.0519 | 8.0540 | 1.0003 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5863 | 5.5861 | 1.0000 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.3712 | 3.3713 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8586 | 10.8618 | 1.0003 | report |  |
| Mouth-box >6 Hz temporal power | 2233.5 | 2236.7 | 1.0014 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.6 / 5.0 | 0.0536 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 4.0 | 0.0607 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 5.0 | 0.0011 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.7195 | 1.7268 | 0.0073 | <= A (1.7195) + 0.05 | PASS |
| Chin positive excess p95 (px) | 3.5430 | 3.5231 | -0.0199 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0617 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1977 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0617 / 0.1977 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 68.95 / face 62.09 | mouth 56.11; worst face 58.72 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9997 | mouth 0.9991; worst mouth 0.9988 | report |  |
| Mouth sharpness (Laplacian var) | 32.9818 | 33.0098 | 1.0009 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 36.1265 | 0.0033 | report |  |
