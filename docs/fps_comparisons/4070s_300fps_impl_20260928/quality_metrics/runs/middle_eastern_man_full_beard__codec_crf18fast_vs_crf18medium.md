### middle_eastern_man_full_beard__codec_crf18fast_vs_crf18medium — INCOMPLETE (profile e1, frames video, 240 frames)

A = `reencode_crf18_fast` (refined), B = `reencode_crf18_medium` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0737 | 0.0732 | 0.9980 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 7.9790 | 7.9186 | 0.2695 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 9.9382 | 9.9354 | 0.9997 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 8.7976 | 8.8493 | 1.0059 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 4.9454 | 4.9719 | 1.0054 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.9391 | 12.9308 | 0.9994 | report |  |
| Mouth-box >6 Hz temporal power | 2862.8 | 2876.6 | 1.0048 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 30.0 / 34.0 | 1.8831 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 23.0 / 26.0 | 2.0839 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 33.0 | 1.2269 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 2.0825 | 2.0850 | 0.0026 | <= A (2.0825) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 1.9876 | 1.7162 | -0.2714 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.3375 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 1.0310 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 42.50 / face 40.06 | mouth 39.90; worst face 38.16 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9790 / face 0.9770 | mouth 0.9804; worst mouth 0.9759 | report |  |
| Mouth sharpness (Laplacian var) | 31.7171 | 31.5730 | 0.9955 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 44.4405 | 44.4261 | 0.0249 | report |  |
