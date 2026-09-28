### white_man_clean_shaven__codec_raw_vs_crf18 — INCOMPLETE (profile e1, frames raw_vs_video, 240 frames)

A = `A_refined_raw` (refined), B = `A_refined_mp4` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0883 | 0.9977 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.6922 | 0.3056 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5045 | 6.4427 | 0.9905 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.2035 | 3.9358 | 0.9363 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3011 | 2.1821 | 0.9483 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5605 | 8.5423 | 0.9979 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1281.3 | 0.9778 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 22.6 / 27.0 | 2.3939 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 20.2 / 27.0 | 2.3983 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 35.0 | 2.1350 | report |  |
| Protected-lip max RGB change vs own standard | 0 | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.1328 | 1.1638 | 0.0310 | <= A (1.1328) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 1.6628 | 1.8524 | 0.1896 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.3017 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.8744 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 39.61 / face 38.42 | mouth 37.92; worst face 36.12 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9766 / face 0.9784 | mouth 0.9852; worst mouth 0.9795 | report |  |
| Mouth sharpness (Laplacian var) | 22.9565 | 31.2730 | 1.3623 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 52.4477 | 0.9130 | report |  |
