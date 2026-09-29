### black_man_short_beard__stagewise16_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `stagewise16_taesdtrt` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1294 | 0.1294 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.1362 | 14.1340 | 0.0617 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.0522 | 8.0531 | 1.0001 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.5868 | 5.5846 | 0.9996 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.3712 | 3.3711 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.8581 | 10.8606 | 1.0002 | report |  |
| Mouth-box >6 Hz temporal power | 2233.5 | 2235.7 | 1.0010 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 3.0 | 0.0524 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0593 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 4.0 | 0.0010 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.7195 | 1.7236 | 0.0041 | <= A (1.7195) + 0.05 | PASS |
| Chin positive excess p95 (px) | 3.5430 | 3.5554 | 0.0124 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0592 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1907 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0592 / 0.1907 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 69.07 / face 62.20 | mouth 56.15; worst face 59.87 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9997 | mouth 0.9991; worst mouth 0.9987 | report |  |
| Mouth sharpness (Laplacian var) | 32.9852 | 33.0023 | 1.0005 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 36.1263 | 36.1265 | 0.0031 | report |  |
