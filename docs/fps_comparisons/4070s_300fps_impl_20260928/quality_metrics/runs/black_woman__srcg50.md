### black_woman__srcg50 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcg50` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0872 | 0.9997 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.7429 | 0.1064 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3182 | 7.3235 | 1.0007 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1251 | 4.1270 | 1.0005 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9755 | 3.9742 | 0.9997 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5291 | 9.5378 | 1.0009 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1716.7 | 1.0015 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 5.0 | 0.0717 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0872 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 6.0 | 0.0011 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 1.6074 | -0.0155 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.3255 | 0.0287 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0861 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2766 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0861 / 0.2766 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 66.89 / face 59.69 | mouth 52.76; worst face 53.70 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9996 | mouth 0.9984; worst mouth 0.9930 | report |  |
| Mouth sharpness (Laplacian var) | 21.3616 | 21.4024 | 1.0019 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2932 | 0.0051 | report |  |
