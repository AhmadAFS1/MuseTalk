### black_woman__srcmix_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcmix_taesdtrt` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0871 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.7410 | 0.0534 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3190 | 7.3171 | 0.9997 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1251 | 4.1262 | 1.0003 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9746 | 3.9737 | 0.9998 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5303 | 9.5271 | 0.9997 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1713.3 | 0.9995 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.6 / 4.0 | 0.0409 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.6 / 4.0 | 0.0510 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 4.0 | 0.0006 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 1.6223 | -0.0005 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.3008 | 0.0040 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0502 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1441 | <= 0.15 | PASS |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0502 / 0.1441 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 70.48 / face 63.30 | mouth 57.02; worst face 58.64 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9998 | mouth 0.9992; worst mouth 0.9982 | report |  |
| Mouth sharpness (Laplacian var) | 21.3665 | 21.3659 | 1.0000 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2932 | 0.0051 | report |  |
