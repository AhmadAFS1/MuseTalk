### south_asian_woman__stagewise16_taesdtrt — PASS (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `stagewise16_taesdtrt` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1053 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.1718 | 0.0432 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2735 | 7.2753 | 1.0003 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.4892 | 4.4892 | 1.0000 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2685 | 3.2686 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7756 | 9.7802 | 1.0005 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 1974.9 | 1.0007 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0359 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0439 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 2.0 | 0.0002 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.3558 | 2.3565 | 0.0007 | <= A (2.3558) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.8213 | 0.8259 | 0.0045 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0357 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1055 | <= 0.15 | PASS |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0357 / 0.1055 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 72.88 / face 64.18 | mouth 57.70; worst face 60.72 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9998 | mouth 0.9993; worst mouth 0.9987 | report |  |
| Mouth sharpness (Laplacian var) | 30.7609 | 30.7947 | 1.0011 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 45.2196 | 0.0036 | report |  |
