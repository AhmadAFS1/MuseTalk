### black_woman__refined_vs_standard_video — FAIL (profile e1, frames video, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0871 | 0.0860 | 0.9932 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7355 | 9.6126 | 0.5491 | <= 0.5 | FAIL |
| Flicker mouth mean abs(dt) RGB | 7.2691 | 7.2438 | 0.9965 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 3.8440 | 3.4358 | 0.8938 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.7549 | 3.6934 | 0.9836 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5259 | 9.5112 | 0.9985 | report |  |
| Mouth-box >6 Hz temporal power | 1682.6 | 1609.4 | 0.9565 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 67.0 / 79.0 | 2.6949 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 65.6 / 79.0 | 3.5224 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 42.0 | 1.1560 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.6703 | 0.7654 | -0.9050 | <= A (1.6703) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 0.1888 | 0.6431 | 0.4543 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.7248 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 2.7197 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 41.55 / face 37.48 | mouth 38.45; worst face 34.32 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9786 / face 0.9722 | mouth 0.9759; worst mouth 0.9555 | report |  |
| Mouth sharpness (Laplacian var) | 29.2713 | 28.0965 | 0.9599 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 41.6054 | 41.5153 | 0.1413 | report |  |
