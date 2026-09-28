### white_man_clean_shaven__codec_crf18fast_vs_crf18medium — INCOMPLETE (profile e1, frames video, 240 frames)

A = `reencode_crf18_fast` (refined), B = `reencode_crf18_medium` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0883 | 0.0885 | 0.9978 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.6922 | 8.7136 | 0.2837 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.4452 | 6.4453 | 1.0000 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 3.9374 | 3.9568 | 1.0049 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.1821 | 2.1784 | 0.9983 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5469 | 8.5262 | 0.9976 | report |  |
| Mouth-box >6 Hz temporal power | 1274.7 | 1279.3 | 1.0036 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 21.6 / 23.0 | 1.4354 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 17.2 / 22.0 | 1.4416 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 29.0 | 1.1690 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.1638 | 1.1622 | -0.0017 | <= A (1.1638) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 1.8524 | 1.7136 | -0.1387 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.2853 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.8468 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 43.34 / face 41.56 | mouth 41.07; worst face 40.30 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9804 / face 0.9799 | mouth 0.9834; worst mouth 0.9787 | report |  |
| Mouth sharpness (Laplacian var) | 31.2930 | 31.1807 | 0.9964 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 52.4477 | 52.4345 | 0.0466 | report |  |
