### south_asian_woman__codec_raw_vs_crf18 — INCOMPLETE (profile e1, frames raw_vs_video, 240 frames)

A = `A_refined_raw` (refined), B = `A_refined_mp4` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1044 | 0.9979 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.0844 | 0.3061 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2686 | 7.2378 | 0.9958 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.4815 | 4.1495 | 0.9259 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2688 | 3.1955 | 0.9776 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7684 | 9.8103 | 1.0043 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 1938.8 | 0.9824 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 25.0 / 29.0 | 2.2843 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 21.0 / 23.0 | 2.3580 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 42.0 | 2.1518 | report |  |
| Protected-lip max RGB change vs own standard | 0 | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 2.3558 | 2.4546 | 0.0988 | <= A (2.3558) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 0.8213 | 0.6806 | -0.1408 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.2595 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.7435 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 39.39 / face 38.46 | mouth 37.84; worst face 36.07 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9706 / face 0.9760 | mouth 0.9842; worst mouth 0.9791 | report |  |
| Mouth sharpness (Laplacian var) | 30.7314 | 39.1949 | 1.2754 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 44.5969 | 0.8229 | report |  |
