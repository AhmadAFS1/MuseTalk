### south_asian_woman__codec_crf18fast_vs_crf18medium — INCOMPLETE (profile e1, frames video, 240 frames)

A = `reencode_crf18_fast` (refined), B = `reencode_crf18_medium` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1044 | 0.1047 | 0.9986 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.0844 | 10.1139 | 0.2462 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2414 | 7.2387 | 0.9996 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1507 | 4.1467 | 0.9990 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1955 | 3.1737 | 0.9932 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.8146 | 9.8019 | 0.9987 | report |  |
| Mouth-box >6 Hz temporal power | 1938.8 | 1941.8 | 1.0015 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 24.0 / 28.0 | 1.4644 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 20.6 / 23.0 | 1.4571 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 27.0 | 1.2908 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 2.4546 | 2.4502 | -0.0044 | <= A (2.4546) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 0.6806 | 0.6962 | 0.0156 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.2378 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.6655 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 42.68 / face 41.33 | mouth 40.70; worst face 40.25 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9778 / face 0.9810 | mouth 0.9817; worst mouth 0.9764 | report |  |
| Mouth sharpness (Laplacian var) | 39.2224 | 39.0330 | 0.9952 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 44.5969 | 44.5829 | 0.0347 | report |  |
