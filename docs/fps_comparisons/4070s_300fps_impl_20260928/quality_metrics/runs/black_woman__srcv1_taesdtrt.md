### black_woman__srcv1_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcv1_taesdtrt` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0873 | 0.9977 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.7627 | 0.3254 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3171 | 7.4392 | 1.0167 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1235 | 4.1282 | 1.0011 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9751 | 4.0019 | 1.0067 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5274 | 9.7817 | 1.0267 | report |  |
| Mouth-box >6 Hz temporal power | 1705.8 | 1745.5 | 1.0232 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 11.0 / 12.0 | 0.3748 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 9.0 / 10.0 | 0.4668 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 20.0 | 0.0038 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 1.6131 | -0.0097 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.3325 | 0.0357 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.2547 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.7787 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.2547 / 0.7787 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 57.60 / face 50.36 | mouth 43.16; worst face 46.44 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9997 / face 0.9983 | mouth 0.9934; worst mouth 0.9798 | report |  |
| Mouth sharpness (Laplacian var) | 21.3644 | 21.3932 | 1.0013 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2858 | 0.0328 | report |  |
