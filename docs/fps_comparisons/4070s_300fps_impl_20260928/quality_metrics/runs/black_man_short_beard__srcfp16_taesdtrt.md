### black_man_short_beard__srcfp16_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcfp16_taesdtrt` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1294 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 14.1351 | 0.0696 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.0516 | 8.0541 | 1.0003 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5869 | 5.5881 | 1.0002 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.3711 | 3.3716 | 1.0001 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8567 | 10.8606 | 1.0004 | report |  |
| Mouth-box >6 Hz temporal power | 2233.5 | 2236.3 | 1.0013 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 6.0 | 0.0531 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0604 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 6.0 | 0.0011 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.7195 | 1.7287 | 0.0092 | <= A (1.7195) + 0.05 | PASS |
| Chin positive excess p95 (px) | 3.5430 | 3.5596 | 0.0166 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0613 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1943 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0613 / 0.1943 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 68.97 / face 62.11 | mouth 56.14; worst face 59.08 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9997 | mouth 0.9991; worst mouth 0.9984 | report |  |
| Mouth sharpness (Laplacian var) | 32.9882 | 33.0097 | 1.0007 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 36.1265 | 0.0032 | report |  |
