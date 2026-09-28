### black_woman__synthetic_faces_noise1lsb_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise1lsb` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0872 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.7534 | 0.0622 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3188 | 7.3655 | 1.0064 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1255 | 4.1308 | 1.0013 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9747 | 3.9996 | 1.0063 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5300 | 9.6335 | 1.0109 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1719.0 | 1.0028 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 6.0 | 0.1898 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.2313 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 7.0 | 0.0010 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 1.6216 | -0.0012 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.2923 | -0.0045 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0622 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1832 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 64.87 / face 57.64 | mouth 51.91; worst face 55.93 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9998 / face 0.9991 | mouth 0.9969; worst mouth 0.9965 | report |  |
| Mouth sharpness (Laplacian var) | 21.3658 | 22.6510 | 1.0601 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2926 | 0.0006 | report |  |
