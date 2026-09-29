### black_man_short_beard__self_video — INCOMPLETE (profile e1, frames video, 240 frames)

A = `A_refined_mp4` (refined), B = `A_refined_mp4_again` (refined); identity `black_man_short_beard`; pre-encode/decoded frames SHA-identical: **True**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1289 | 0.1289 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 14.0845 | 14.0845 | 0.0000 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.9327 | 7.9327 | 1.0000 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 5.1446 | 5.1446 | 1.0000 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1370 | 3.1370 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 10.7380 | 10.7380 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 2186.7 | 2186.7 | 1.0000 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 0.0 | 0.0000 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.7268 | 1.7268 | 0.0000 | <= A (1.7268) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 3.5781 | 3.5781 | 0.0000 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0000 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.0000 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 100.00 / face 100.00 | mouth 100.00; worst face 100.00 | report-only | REPORT |
| SSIM A vs B mean | - | full 1.0000 / face 1.0000 | mouth 1.0000; worst mouth 1.0000 | report |  |
| Mouth sharpness (Laplacian var) | 41.6858 | 41.6858 | 1.0000 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 35.4243 | 35.4243 | 0.0000 | report |  |
