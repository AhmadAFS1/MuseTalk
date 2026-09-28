### black_woman__codec_crf18fast_vs_crf18medium — INCOMPLETE (profile e1, frames video, 240 frames)

A = `reencode_crf18_fast` (refined), B = `reencode_crf18_medium` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0871 | 0.0875 | 0.9983 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7355 | 9.7867 | 0.2736 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2736 | 7.2721 | 0.9998 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 3.8655 | 3.8580 | 0.9981 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.8953 | 3.8823 | 0.9967 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5338 | 9.5093 | 0.9974 | report |  |
| Mouth-box >6 Hz temporal power | 1674.2 | 1680.1 | 1.0035 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 24.0 / 27.0 | 1.4974 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 19.0 / 23.0 | 1.5189 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 31.0 | 1.2555 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.6703 | 1.7063 | 0.0359 | <= A (1.6703) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 0.1888 | 0.2524 | 0.0635 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.2883 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.7565 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 42.70 / face 41.06 | mouth 40.82; worst face 39.39 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9788 / face 0.9786 | mouth 0.9812; worst mouth 0.9769 | report |  |
| Mouth sharpness (Laplacian var) | 29.2985 | 28.8858 | 0.9859 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 41.6054 | 41.5940 | 0.0435 | report |  |
