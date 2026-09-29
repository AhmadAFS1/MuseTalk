### south_asian_woman__self_video — INCOMPLETE (profile e1, frames video, 240 frames)

A = `A_refined_mp4` (refined), B = `A_refined_mp4_again` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **True**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1044 | 0.1044 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.0844 | 10.0844 | 0.0000 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2455 | 7.2455 | 1.0000 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1611 | 4.1611 | 1.0000 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1955 | 3.1955 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.8218 | 9.8218 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 1938.8 | 1938.8 | 1.0000 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 0.0 | 0.0000 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 2.4546 | 2.4546 | 0.0000 | <= A (2.4546) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 0.6806 | 0.6806 | 0.0000 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0000 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.0000 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 100.00 / face 100.00 | mouth 100.00; worst face 100.00 | report-only | REPORT |
| SSIM A vs B mean | - | full 1.0000 / face 1.0000 | mouth 1.0000; worst mouth 1.0000 | report |  |
| Mouth sharpness (Laplacian var) | 39.2641 | 39.2641 | 1.0000 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 44.5969 | 44.5969 | 0.0000 | report |  |
