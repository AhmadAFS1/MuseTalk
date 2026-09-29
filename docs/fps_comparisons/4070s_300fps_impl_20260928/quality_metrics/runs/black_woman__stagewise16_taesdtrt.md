### black_woman__stagewise16_taesdtrt — PASS (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `stagewise16_taesdtrt` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0872 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.7446 | 0.0522 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3188 | 7.3197 | 1.0001 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1256 | 4.1270 | 1.0003 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9753 | 3.9754 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5303 | 9.5318 | 1.0002 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1714.6 | 1.0002 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0412 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0511 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 5.0 | 0.0007 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 1.6218 | -0.0010 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.2810 | -0.0158 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0492 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1392 | <= 0.15 | PASS |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0492 / 0.1392 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 70.45 / face 63.27 | mouth 57.06; worst face 60.36 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9998 | mouth 0.9992; worst mouth 0.9986 | report |  |
| Mouth sharpness (Laplacian var) | 21.3654 | 21.3729 | 1.0003 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2932 | 0.0050 | report |  |
