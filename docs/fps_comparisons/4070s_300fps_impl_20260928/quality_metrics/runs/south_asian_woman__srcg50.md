### south_asian_woman__srcg50 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcg50` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1053 | 0.9998 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.1761 | 0.0872 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2727 | 7.2808 | 1.0011 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.4891 | 4.4866 | 0.9994 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2685 | 3.2680 | 0.9998 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7745 | 9.7914 | 1.0017 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 1983.0 | 1.0048 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 4.0 | 0.0580 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.6 / 4.0 | 0.0710 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 4.0 | 0.0003 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.3558 | 2.3443 | -0.0115 | <= A (2.3558) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.8213 | 0.8480 | 0.0267 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0640 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2030 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0640 / 0.2030 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 69.39 / face 60.69 | mouth 53.38; worst face 55.66 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9997 | mouth 0.9987; worst mouth 0.9974 | report |  |
| Mouth sharpness (Laplacian var) | 30.7565 | 30.8550 | 1.0032 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 45.2200 | 0.0035 | report |  |
