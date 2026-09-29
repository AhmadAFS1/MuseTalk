### japanese__int8_vs_taesd_video — FAIL (profile lossy, frames video, 240 frames)

A = `approved_INT8_chin100` (source), B = `TAESD_chin100` (source); identity `japanese`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0992 | 0.0960 | 0.9925 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.5548 | 10.2118 | 0.6499 | <= 0.5 | FAIL |
| Flicker mouth mean abs(dt) RGB | 6.3371 | 6.0194 | 0.9499 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 3.8908 | 3.8816 | 0.9976 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.6149 | 2.5782 | 0.9860 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5426 | 8.0820 | 0.9461 | report |  |
| Mouth-box >6 Hz temporal power | 1837.8 | 1645.2 | 0.8952 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 26.6 / 32.0 | 1.7467 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 25.8 / 32.0 | 1.6609 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 27.0 | 1.0751 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.1285 | 1.0675 | -0.0610 | <= A (1.1285) + 0.2 | PASS |
| Chin positive excess p95 (px) | 1.9618 | 1.9002 | -0.0616 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.4725 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 1.3814 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 42.97 / face 39.65 | mouth 34.68; worst face 38.39 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9826 / face 0.9803 | mouth 0.9724; worst mouth 0.9604 | report |  |
| Mouth sharpness (Laplacian var) | 44.6926 | 37.5049 | 0.8392 | >= 0.95 | FAIL |
| Face Lab L* mean; delta = dE76 of means | 48.8442 | 48.8122 | 0.3170 | report |  |
