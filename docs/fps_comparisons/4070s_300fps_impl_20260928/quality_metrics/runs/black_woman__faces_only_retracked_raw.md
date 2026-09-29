### black_woman__faces_only_retracked_raw — PASS (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_only` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **True**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0872 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.7519 | 0.0000 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3190 | 7.3190 | 1.0000 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1264 | 4.1264 | 1.0000 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9746 | 3.9746 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5303 | 9.5303 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1714.2 | 1.0000 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 0.0 | 0.0000 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 1.6228 | 0.0000 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.2968 | 0.0000 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0000 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.0000 | <= 0.15 | PASS |
| PSNR A vs B, dB mean frame (cap 100) | - | full 100.00 / face 100.00 | mouth 100.00; worst face 100.00 | report-only | REPORT |
| SSIM A vs B mean | - | full 1.0000 / face 1.0000 | mouth 1.0000; worst mouth 1.0000 | report |  |
| Mouth sharpness (Laplacian var) | 21.3665 | 21.3665 | 1.0000 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2931 | 0.0000 | report |  |
