### east_asian_man_goatee__codec_crf18fast_vs_crf18medium — INCOMPLETE (profile e1, frames video, 240 frames)

A = `reencode_crf18_fast` (refined), B = `reencode_crf18_medium` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0874 | 0.9985 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.9433 | 8.9634 | 0.2335 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.8548 | 8.8565 | 1.0002 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.9794 | 5.9929 | 1.0022 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.0917 | 3.0767 | 0.9952 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.0457 | 12.0237 | 0.9982 | report |  |
| Mouth-box >6 Hz temporal power | 2541.5 | 2549.1 | 1.0030 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 23.6 / 29.0 | 1.4916 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 22.2 / 29.0 | 1.5044 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 37.0 | 1.1806 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 2.6824 | 2.6495 | -0.0329 | <= A (2.6824) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 0.1106 | 0.1182 | 0.0076 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.2388 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.6541 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 43.04 / face 41.14 | mouth 40.22; worst face 39.85 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9799 / face 0.9796 | mouth 0.9824; worst mouth 0.9782 | report |  |
| Mouth sharpness (Laplacian var) | 36.9580 | 36.5850 | 0.9899 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.2110 | 54.1987 | 0.0329 | report |  |
