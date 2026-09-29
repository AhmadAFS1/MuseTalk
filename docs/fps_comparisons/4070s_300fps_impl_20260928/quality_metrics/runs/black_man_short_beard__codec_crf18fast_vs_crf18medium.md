### black_man_short_beard__codec_crf18fast_vs_crf18medium — INCOMPLETE (profile e1, frames video, 240 frames)

A = `reencode_crf18_fast` (refined), B = `reencode_crf18_medium` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1289 | 0.1291 | 0.9980 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.0845 | 14.1028 | 0.3686 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.9270 | 7.9282 | 1.0002 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.1395 | 5.1743 | 1.0068 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1370 | 3.1340 | 0.9991 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.7283 | 10.7034 | 0.9977 | report |  |
| Mouth-box >6 Hz temporal power | 2180.6 | 2185.4 | 1.0022 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 22.6 / 24.0 | 1.6926 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 20.0 / 24.0 | 1.7387 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 33.0 | 1.1779 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.7268 | 1.7074 | -0.0194 | <= A (1.7268) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 3.5781 | 3.6143 | 0.0362 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.3237 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.9652 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 43.04 / face 40.78 | mouth 40.27; worst face 39.08 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9787 / face 0.9763 | mouth 0.9805; worst mouth 0.9768 | report |  |
| Mouth sharpness (Laplacian var) | 41.6411 | 41.1538 | 0.9883 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 35.4243 | 35.4078 | 0.0336 | report |  |
