### black_woman__self_video — INCOMPLETE (profile e1, frames video, 240 frames)

A = `A_refined_mp4` (refined), B = `A_refined_mp4_again` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **True**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0871 | 0.0871 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7355 | 9.7355 | 0.0000 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2781 | 7.2781 | 1.0000 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 3.8689 | 3.8689 | 1.0000 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.8953 | 3.8953 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5412 | 9.5412 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 1682.6 | 1682.6 | 1.0000 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 0.0 | 0.0000 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.6703 | 1.6703 | 0.0000 | <= A (1.6703) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 0.1888 | 0.1888 | 0.0000 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0000 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.0000 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 100.00 / face 100.00 | mouth 100.00; worst face 100.00 | report-only | REPORT |
| SSIM A vs B mean | - | full 1.0000 / face 1.0000 | mouth 1.0000; worst mouth 1.0000 | report |  |
| Mouth sharpness (Laplacian var) | 29.3194 | 29.3194 | 1.0000 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 41.6054 | 41.6054 | 0.0000 | report |  |
