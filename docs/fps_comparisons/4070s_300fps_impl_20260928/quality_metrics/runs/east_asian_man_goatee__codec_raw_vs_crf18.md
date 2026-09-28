### east_asian_man_goatee__codec_raw_vs_crf18 — INCOMPLETE (profile e1, frames raw_vs_video, 240 frames)

A = `A_refined_raw` (refined), B = `A_refined_mp4` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0872 | 0.9986 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 8.9433 | 0.2360 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9432 | 8.8520 | 0.9898 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3059 | 5.9784 | 0.9481 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1932 | 3.0917 | 0.9682 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1115 | 12.0406 | 0.9941 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2541.5 | 0.9805 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 27.6 / 32.0 | 2.4055 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 26.2 / 30.0 | 2.4328 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 37.0 | 2.1820 | report |  |
| Protected-lip max RGB change vs own standard | 0 | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 2.6024 | 2.6824 | 0.0800 | <= A (2.6024) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 0.1348 | 0.1106 | -0.0242 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.2553 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.6949 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 39.43 / face 38.20 | mouth 37.49; worst face 35.77 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9768 / face 0.9780 | mouth 0.9848; worst mouth 0.9798 | report |  |
| Mouth sharpness (Laplacian var) | 26.3362 | 36.9280 | 1.4022 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.2110 | 0.9069 | report |  |
